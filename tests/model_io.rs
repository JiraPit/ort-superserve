//! Models with several named inputs and outputs of any element type, per-output
//! splitting, fixed batch sizes and variable-length inputs.

mod common;

use anyhow::Result;
use common::{ModelFixture, env, onnx};
use futures::future::join_all;
use ndarray::{Array1, array};
use ort_superserve::{
    Element, Input, InputSpec, Inputs, Output, OutputSpec, Outputs, Server, ServerConfig, Split,
    split_batch_axis,
};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Records the number of requests in every batch.
#[derive(Clone, Default)]
struct BatchLog(Arc<Mutex<Vec<usize>>>);

impl BatchLog {
    fn sizes(&self) -> Vec<usize> {
        self.0.lock().unwrap().clone()
    }
}

/// A request carrying ready-made model inputs.
struct Request {
    inputs: Inputs,
    log: BatchLog,
}

fn request(inputs: impl Into<Inputs>) -> Request {
    Request {
        inputs: inputs.into(),
        log: BatchLog::default(),
    }
}

impl Request {
    fn logged(mut self, log: &BatchLog) -> Self {
        self.log = log.clone();
        self
    }
}

struct Prepared {
    inputs: Inputs,
    log: BatchLog,
}

impl From<Prepared> for Inputs {
    fn from(prepared: Prepared) -> Self {
        prepared.inputs
    }
}

impl Input for Request {
    type Preprocessed = Prepared;
    type Context = ();

    async fn preprocess(self) -> Result<(Prepared, ())> {
        Ok((
            Prepared {
                inputs: self.inputs,
                log: self.log,
            },
            (),
        ))
    }

    fn batch(items: Vec<Prepared>) -> Result<Vec<Inputs>> {
        if let Some(first) = items.first() {
            first.log.0.lock().unwrap().push(items.len());
        }
        Ok(items.into_iter().map(Inputs::from).collect())
    }
}

/// One request's outputs, as handed to postprocessing.
#[derive(Debug)]
struct Raw(Outputs);

impl Output for Raw {
    async fn postprocess(outputs: Outputs, _context: ()) -> Result<Self> {
        Ok(Raw(outputs))
    }
}

impl Raw {
    fn values<T: Element>(&self, name: &str) -> Vec<T> {
        self.0.get::<T>(name).unwrap().iter().copied().collect()
    }

    fn shape(&self, name: &str) -> Vec<usize> {
        self.0.get::<f32>(name).unwrap().shape().to_vec()
    }

    fn names(&self) -> Vec<&str> {
        self.0.names().collect()
    }
}

async fn try_serve(model: Vec<u8>, config: ServerConfig) -> Result<Server<Request, Raw>> {
    let model = ModelFixture::new(model);
    Server::from_file(model.path(), config.with_environment(env())).await
}

async fn serve(model: Vec<u8>, config: ServerConfig) -> Server<Request, Raw> {
    try_serve(model, config)
        .await
        .expect("Failed to start server")
}

/// Config whose batching window is long enough that concurrently submitted
/// requests are guaranteed to land in the same batch.
fn single_batch_config(max_batch_size: usize) -> ServerConfig {
    ServerConfig::new()
        .with_max_batch_size(max_batch_size)
        .with_max_wait_time(Duration::from_secs(2))
}

fn startup_error(result: Result<Server<Request, Raw>>) -> String {
    match result {
        Ok(_) => panic!("expected the server to fail to start"),
        Err(e) => format!("{e:#}"),
    }
}

fn error_message(result: Result<Raw>) -> String {
    format!("{:#}", result.expect_err("expected inference to fail"))
}

/// Inputs of the token models.
fn tokens(ids: &[i64], mask: &[i64]) -> Inputs {
    Inputs::new()
        .with("input_ids", Array1::from(ids.to_vec()))
        .with("attention_mask", Array1::from(mask.to_vec()))
}

/// Inputs of the token models with every position attended.
fn sequence(ids: &[i64]) -> Inputs {
    tokens(ids, &vec![1; ids.len()])
}

mod named_inputs_and_outputs {
    use super::*;

    #[tokio::test]
    async fn every_output_reaches_its_own_caller_with_its_element_type() {
        let server = serve(onnx::tokens_model(), single_batch_config(2)).await;
        let log = BatchLog::default();

        let (a, b) = tokio::join!(
            server.infer(request(tokens(&[1, 2], &[1, 0])).logged(&log)),
            server.infer(request(tokens(&[3, 4], &[0, 1])).logged(&log)),
        );
        let (a, b) = (a.unwrap(), b.unwrap());

        assert_eq!(log.sizes(), vec![2]);
        assert_eq!(a.names(), ["masked", "labels", "max_id"]);
        assert_eq!(a.values::<f32>("masked"), [1.0, 0.0]);
        assert_eq!(a.values::<i64>("labels"), [1, 2]);
        assert_eq!(a.values::<f32>("max_id"), [2.0]);
        assert_eq!(b.values::<f32>("masked"), [0.0, 4.0]);
        assert_eq!(b.values::<i64>("labels"), [3, 4]);
        assert_eq!(b.values::<f32>("max_id"), [4.0]);
    }

    #[tokio::test]
    async fn inputs_and_outputs_of_other_element_types_are_served() {
        let server = serve(onnx::element_types_model(), single_batch_config(2)).await;

        let (a, b) = tokio::join!(
            server.infer(request(array![0u8, 1, 2])),
            server.infer(request(array![3u8, 0, 255])),
        );
        let (a, b) = (a.unwrap(), b.unwrap());

        assert_eq!(a.values::<f64>("as_f64"), [0.0, 1.0, 2.0]);
        assert_eq!(a.values::<bool>("nonzero"), [false, true, true]);
        assert_eq!(b.values::<f64>("as_f64"), [3.0, 0.0, 255.0]);
        assert_eq!(b.values::<bool>("nonzero"), [true, false, true]);
    }

    #[tokio::test]
    async fn reading_an_output_as_another_element_type_fails() {
        let server = serve(onnx::tokens_model(), single_batch_config(1)).await;

        let output = server.infer(request(sequence(&[1, 2]))).await.unwrap();

        assert!(output.0.get::<f32>("labels").is_err());
        assert!(output.0.get::<i64>("missing").is_err());
    }

    #[tokio::test]
    async fn only_selected_outputs_are_returned_in_the_configured_order() {
        let config = single_batch_config(1)
            .with_output(OutputSpec::new("max_id"))
            .with_output(OutputSpec::new("labels"));
        let server = serve(onnx::tokens_model(), config).await;

        let output = server.infer(request(sequence(&[5, 6]))).await.unwrap();

        assert_eq!(output.names(), ["max_id", "labels"]);
        assert_eq!(output.values::<f32>("max_id"), [6.0]);
        assert_eq!(output.values::<i64>("labels"), [5, 6]);
    }

    #[tokio::test]
    async fn an_unbatched_input_is_shared_by_the_whole_batch() {
        let server = serve(onnx::scaled_model(), single_batch_config(2)).await;
        let scaled = |x: [f32; 4]| {
            request(
                Inputs::new()
                    .with("x", Array1::from(x.to_vec()))
                    .with("scale", ndarray::arr0(3.0f32)),
            )
        };

        let (a, b) = tokio::join!(
            server.infer(scaled([1.0, 2.0, 3.0, 4.0])),
            server.infer(scaled([5.0, 6.0, 7.0, 8.0])),
        );

        assert_eq!(a.unwrap().values::<f32>("y"), [3.0, 6.0, 9.0, 12.0]);
        assert_eq!(b.unwrap().values::<f32>("y"), [15.0, 18.0, 21.0, 24.0]);
    }

    #[tokio::test]
    async fn different_values_for_an_unbatched_input_fail_their_batch() {
        let server = serve(onnx::scaled_model(), single_batch_config(2)).await;
        let scaled = |scale: f32| {
            request(
                Inputs::new()
                    .with("x", array![1.0f32, 2.0, 3.0, 4.0])
                    .with("scale", ndarray::arr0(scale)),
            )
        };

        let (a, b) = tokio::join!(server.infer(scaled(2.0)), server.infer(scaled(3.0)));

        assert!(error_message(a).contains("'scale' is not batched"));
        assert!(error_message(b).contains("'scale' is not batched"));
    }

    #[tokio::test]
    async fn a_malformed_request_fails_alone() {
        let server = serve(onnx::tokens_model(), single_batch_config(6)).await;
        let log = BatchLog::default();
        let ids = Array1::from(vec![1i64, 2]);

        let results = join_all(
            [
                sequence(&[1, 2]),
                Inputs::new().with("input_ids", ids.clone()),
                Inputs::new()
                    .with("input_ids", array![1.0f32, 2.0])
                    .with("attention_mask", array![1i64, 1]),
                sequence(&[1, 2]).with("token_type_ids", ids.clone()),
                sequence(&[1, 2]).with("input_ids", ids.clone()),
                tokens(&[1, 2, 3], &[1, 1]),
            ]
            .into_iter()
            .map(|inputs| server.infer(request(inputs).logged(&log))),
        )
        .await;
        let mut results = results.into_iter();

        assert_eq!(
            results.next().unwrap().unwrap().values::<i64>("labels"),
            [1, 2]
        );
        let expected = [
            "missing model input 'attention_mask'",
            "input 'input_ids' must have element type i64",
            "the model has no input named 'token_type_ids'",
            "input 'input_ids' was given more than once",
            "must have the same length along dimension 'sequence'",
        ];
        for (result, expected) in results.zip(expected) {
            let message = error_message(result);
            assert!(message.contains(expected), "{message}");
        }
        assert_eq!(log.sizes(), vec![6]);
    }

    #[tokio::test]
    async fn an_input_spec_for_an_unknown_input_fails_startup() {
        let config = single_batch_config(1).with_input(InputSpec::new("token_type_ids"));

        let message = startup_error(try_serve(onnx::tokens_model(), config).await);

        assert!(message.contains("'token_type_ids' does not match any model input"));
    }
}

mod output_splits {
    use super::*;

    /// `x` for request `i` of the NMS model.
    fn nms_input(i: usize) -> Request {
        let i = i as f32;
        request(array![2.0 * i + 1.0, 2.0 * i + 2.0])
    }

    #[tokio::test]
    async fn rows_split_by_a_batch_index_column_reach_their_callers() {
        let config = single_batch_config(3)
            .with_output(OutputSpec::new("detections").with_split(Split::batch_index_column(0)));
        let server = serve(onnx::nms_model(), config).await;

        let results = join_all((0..3).map(|i| server.infer(nms_input(i)))).await;

        for (i, result) in results.into_iter().enumerate() {
            let output = result.unwrap();
            let (b, x0, x1) = (i as f32, 2.0 * i as f32 + 1.0, 2.0 * i as f32 + 2.0);
            assert_eq!(output.shape("detections"), [2, 3]);
            assert_eq!(
                output.values::<f32>("detections"),
                [b, x0, x1, b, x0 + 10.0, x1 + 10.0]
            );
        }
    }

    #[tokio::test]
    async fn rows_split_by_another_outputs_batch_index_reach_their_callers() {
        let config = single_batch_config(3)
            .with_output_split("boxes", Split::batch_index_from("batch_index", None));
        let server = serve(onnx::nms_model(), config).await;

        let results = join_all((0..3).map(|i| server.infer(nms_input(i)))).await;

        for (i, result) in results.into_iter().enumerate() {
            let output = result.unwrap();
            let (x0, x1) = (2.0 * i as f32 + 1.0, 2.0 * i as f32 + 2.0);
            assert_eq!(output.names(), ["boxes"]);
            assert_eq!(
                output.values::<f32>("boxes"),
                [x0, x1, x0 + 10.0, x1 + 10.0]
            );
        }
    }

    #[tokio::test]
    async fn default_split_rejects_an_output_with_more_rows_than_the_batch() {
        let server = serve(onnx::nms_model(), single_batch_config(2)).await;

        let (a, b) = tokio::join!(server.infer(nms_input(0)), server.infer(nms_input(1)));

        assert!(error_message(a).contains("Output count mismatch"));
        assert!(error_message(b).contains("Output count mismatch"));
    }

    #[tokio::test]
    async fn default_split_rejects_an_output_whose_axis_0_is_another_dimension() {
        let server = serve(onnx::renamed_batch_axis_model(), single_batch_config(2)).await;

        let (a, b) = tokio::join!(
            server.infer(request(array![1.0f32, 2.0, 3.0, 4.0])),
            server.infer(request(array![5.0f32, 6.0, 7.0, 8.0])),
        );

        assert!(error_message(a).contains("not the inputs' batch dimension"));
        assert!(error_message(b).contains("not the inputs' batch dimension"));
    }

    #[tokio::test]
    async fn explicit_batch_axis_split_accepts_an_output_named_differently() {
        let config = single_batch_config(2).with_output_split("y", Split::BatchAxis);
        let server = serve(onnx::renamed_batch_axis_model(), config).await;

        let (a, b) = tokio::join!(
            server.infer(request(array![1.0f32, 2.0, 3.0, 4.0])),
            server.infer(request(array![5.0f32, 6.0, 7.0, 8.0])),
        );

        assert_eq!(a.unwrap().values::<f32>("y"), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(b.unwrap().values::<f32>("y"), [5.0, 6.0, 7.0, 8.0]);
    }

    #[tokio::test]
    async fn custom_split_decides_what_each_caller_receives() {
        let config = single_batch_config(2).with_output_split(
            "y",
            Split::custom(|output, ctx| Ok(vec![output.clone(); ctx.num_requests()])),
        );
        let server = serve(onnx::batch_collapsing_model(), config).await;

        let (a, b) = tokio::join!(
            server.infer(request(array![1.0f32, 2.0, 3.0, 4.0])),
            server.infer(request(array![3.0f32, 4.0, 5.0, 6.0])),
        );

        for output in [a.unwrap(), b.unwrap()] {
            assert_eq!(output.shape("y"), [1, 4]);
            assert_eq!(output.values::<f32>("y"), [2.0, 3.0, 4.0, 5.0]);
        }
    }

    #[tokio::test]
    async fn custom_split_returning_the_wrong_number_of_parts_fails_the_batch() {
        let config = single_batch_config(2)
            .with_output_split("y", Split::custom(|output, _ctx| Ok(vec![output.clone()])));
        let server = serve(onnx::batch_collapsing_model(), config).await;

        let (a, b) = tokio::join!(
            server.infer(request(array![1.0f32, 2.0, 3.0, 4.0])),
            server.infer(request(array![3.0f32, 4.0, 5.0, 6.0])),
        );

        assert!(error_message(a).contains("produced 1 parts for 2 requests"));
        assert!(error_message(b).contains("produced 1 parts for 2 requests"));
    }
}

mod fixed_batch_size {
    use super::*;

    fn row(start: f32) -> Request {
        request(Array1::from_iter((0..4).map(|i| start + i as f32)))
    }

    fn doubled_row(start: f32) -> Vec<f32> {
        (0..4).map(|i| (start + i as f32) * 2.0).collect()
    }

    #[tokio::test]
    async fn partial_batch_is_padded_and_padded_rows_reach_no_caller() {
        let server = serve(onnx::fixed_batch_doubling_model(4), single_batch_config(4)).await;
        let log = BatchLog::default();

        let results =
            join_all((0..3).map(|i| server.infer(row(i as f32 * 10.0).logged(&log)))).await;

        assert_eq!(log.sizes(), vec![3]);
        for (i, result) in results.into_iter().enumerate() {
            let output = result.unwrap();
            assert_eq!(output.shape("y"), [4]);
            assert_eq!(output.values::<f32>("y"), doubled_row(i as f32 * 10.0));
        }
    }

    #[tokio::test]
    async fn batches_are_capped_at_the_model_batch_size() {
        let config = ServerConfig::new()
            .with_max_batch_size(8)
            .with_max_wait_time(Duration::from_millis(200));
        let server = serve(onnx::fixed_batch_doubling_model(4), config).await;
        let log = BatchLog::default();

        let results = join_all((0..6).map(|i| server.infer(row(i as f32).logged(&log)))).await;

        for (i, result) in results.into_iter().enumerate() {
            assert_eq!(result.unwrap().values::<f32>("y"), doubled_row(i as f32));
        }
        let sizes = log.sizes();
        assert!(sizes.iter().all(|&size| size <= 4), "{sizes:?}");
        assert_eq!(sizes.iter().sum::<usize>(), 6);
    }

    #[tokio::test]
    async fn a_batch_size_1_export_serves_concurrent_requests() {
        let server = serve(onnx::fixed_batch_doubling_model(1), single_batch_config(4)).await;
        let log = BatchLog::default();

        let results = join_all((0..3).map(|i| server.infer(row(i as f32).logged(&log)))).await;

        for (i, result) in results.into_iter().enumerate() {
            assert_eq!(result.unwrap().values::<f32>("y"), doubled_row(i as f32));
        }
        assert_eq!(log.sizes(), vec![1, 1, 1]);
    }

    #[tokio::test]
    async fn padded_rows_are_filled_with_the_inputs_pad_value() {
        let broadcast =
            || Split::custom(|output, ctx| Ok(vec![output.clone(); ctx.num_requests()]));
        let column_max = |config: ServerConfig| async move {
            let server = serve(
                onnx::fixed_batch_column_max_model(),
                config.with_output_split("y", broadcast()),
            )
            .await;
            server.infer(row(1.0)).await.unwrap().values::<f32>("y")
        };

        // The model's maximum over all four rows sees the three padded rows.
        assert_eq!(
            column_max(single_batch_config(4)).await,
            [1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            column_max(single_batch_config(4).with_pad_value("x", 100.0)).await,
            [100.0; 4]
        );
    }

    #[tokio::test]
    async fn rows_of_padded_slots_are_dropped_from_a_batch_index_split() {
        let config =
            single_batch_config(4).with_output_split("detections", Split::batch_index_column(0));
        let server = serve(onnx::fixed_batch_nms_model(), config).await;

        let results = join_all(
            [[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]
                .into_iter()
                .map(|x| server.infer(request(Array1::from(x.to_vec())))),
        )
        .await;

        let detections: Vec<Vec<f32>> = results
            .into_iter()
            .map(|r| r.unwrap().values::<f32>("detections"))
            .collect();
        assert_eq!(
            detections,
            [[0.0, 1.0, 2.0], [1.0, 3.0, 4.0], [2.0, 5.0, 6.0]]
        );
    }

    #[tokio::test]
    async fn an_output_with_a_data_dependent_axis_0_is_rejected() {
        let server = serve(onnx::fixed_batch_nonzero_model(), single_batch_config(4)).await;

        let result = server.infer(row(1.0)).await;

        assert!(error_message(result).contains("data-dependent"));
    }

    #[tokio::test]
    async fn inputs_with_different_fixed_batch_sizes_fail_startup() {
        let result = try_serve(
            onnx::conflicting_fixed_batch_model(),
            single_batch_config(4),
        )
        .await;

        assert!(startup_error(result).contains("different fixed batch sizes (2 and 3)"));
    }
}

mod variable_length {
    use super::*;

    #[tokio::test]
    async fn requests_of_different_lengths_share_a_batch_and_get_only_their_positions() {
        let server = serve(onnx::tokens_model(), single_batch_config(3)).await;
        let log = BatchLog::default();

        let (a, b, c) = tokio::join!(
            server.infer(request(tokens(&[1, 2], &[1, 1])).logged(&log)),
            server.infer(request(tokens(&[3, 4, 5, 6], &[1, 1, 0, 1])).logged(&log)),
            server.infer(request(tokens(&[7], &[1])).logged(&log)),
        );
        let (a, b, c) = (a.unwrap(), b.unwrap(), c.unwrap());

        assert_eq!(log.sizes(), vec![3]);
        assert_eq!(a.values::<f32>("masked"), [1.0, 2.0]);
        assert_eq!(a.values::<i64>("labels"), [1, 2]);
        assert_eq!(a.values::<f32>("max_id"), [2.0]);
        assert_eq!(b.values::<f32>("masked"), [3.0, 4.0, 0.0, 6.0]);
        assert_eq!(b.values::<i64>("labels"), [3, 4, 5, 6]);
        assert_eq!(b.values::<f32>("max_id"), [6.0]);
        assert_eq!(c.values::<f32>("masked"), [7.0]);
        assert_eq!(c.values::<i64>("labels"), [7]);
    }

    #[tokio::test]
    async fn padding_value_is_configurable_per_input() {
        let max_id_of_short_request = |config: ServerConfig| async move {
            let server = serve(onnx::tokens_model(), config).await;
            let (short, _long) = tokio::join!(
                server.infer(request(sequence(&[-5, -7]))),
                server.infer(request(sequence(&[-1, -2, -3, -4]))),
            );
            short.unwrap().values::<f32>("max_id")
        };

        // `max_id` is taken over every position the model sees, padding included.
        assert_eq!(max_id_of_short_request(single_batch_config(2)).await, [0.0]);
        assert_eq!(
            max_id_of_short_request(single_batch_config(2).with_pad_value("input_ids", -100.0))
                .await,
            [-5.0]
        );
    }

    #[tokio::test]
    async fn different_lengths_work_together_with_fixed_batch_padding() {
        let server = serve(onnx::fixed_batch_tokens_model(), single_batch_config(4)).await;
        let log = BatchLog::default();

        let (a, b, c) = tokio::join!(
            server.infer(request(tokens(&[1, 2], &[1, 1])).logged(&log)),
            server.infer(request(tokens(&[3, 4, 5, 6], &[1, 1, 0, 1])).logged(&log)),
            server.infer(request(tokens(&[7], &[1])).logged(&log)),
        );

        assert_eq!(log.sizes(), vec![3]);
        assert_eq!(a.unwrap().values::<f32>("masked"), [1.0, 2.0]);
        assert_eq!(b.unwrap().values::<f32>("masked"), [3.0, 4.0, 0.0, 6.0]);
        assert_eq!(c.unwrap().values::<i64>("labels"), [7]);
    }

    #[tokio::test]
    async fn an_output_axis_not_linked_to_a_padded_dimension_fails_startup() {
        let message =
            startup_error(try_serve(onnx::unlinked_sequence_model(), single_batch_config(2)).await);

        assert!(
            message.contains("Output 'y' has dynamic axes [1]"),
            "{message}"
        );
        assert!(message.contains("'x' axis 1"), "{message}");
        assert!(message.contains("with_crop_axis(1, "), "{message}");
        assert!(message.contains("with_uncropped_axis(1)"), "{message}");
    }

    #[tokio::test]
    async fn an_unlinked_output_axis_is_not_checked_for_a_custom_split() {
        let config = single_batch_config(2).with_output_split("y", Split::custom(split_batch_axis));
        let server = serve(onnx::unlinked_sequence_model(), config).await;

        let (a, b) = tokio::join!(
            server.infer(request(array![1.0f32, 2.0])),
            server.infer(request(array![3.0f32, 4.0, 5.0])),
        );

        assert_eq!(a.unwrap().values::<f32>("y"), [2.0, 4.0, 0.0]);
        assert_eq!(b.unwrap().values::<f32>("y"), [6.0, 8.0, 10.0]);
    }

    #[tokio::test]
    async fn an_output_without_a_declared_shape_fails_batches_where_it_has_a_padded_length() {
        let server = serve(
            onnx::unshaped_sequence_output_model(),
            single_batch_config(2),
        )
        .await;

        let (a, b) = tokio::join!(
            server.infer(request(array![1.0f32, 2.0])),
            server.infer(request(array![3.0f32, 4.0, 5.0])),
        );
        assert!(error_message(a).contains("with_crop_axis"));
        assert!(error_message(b).contains("with_crop_axis"));

        let (c, d) = tokio::join!(
            server.infer(request(array![1.0f32, 2.0])),
            server.infer(request(array![3.0f32, 4.0])),
        );
        assert_eq!(c.unwrap().values::<f32>("y"), [2.0, 4.0]);
        assert_eq!(d.unwrap().values::<f32>("y"), [6.0, 8.0]);
    }

    #[tokio::test]
    async fn a_declared_crop_axis_is_cut_back_per_request() {
        let config =
            single_batch_config(2).with_output(OutputSpec::new("y").with_crop_axis(1, "x", 1));
        let server = serve(onnx::unlinked_sequence_model(), config).await;

        let (a, b) = tokio::join!(
            server.infer(request(array![1.0f32, 2.0])),
            server.infer(request(array![3.0f32, 4.0, 5.0])),
        );

        assert_eq!(a.unwrap().values::<f32>("y"), [2.0, 4.0]);
        assert_eq!(b.unwrap().values::<f32>("y"), [6.0, 8.0, 10.0]);
    }

    #[tokio::test]
    async fn a_declared_uncropped_axis_keeps_its_full_length() {
        let config =
            single_batch_config(2).with_output(OutputSpec::new("y").with_uncropped_axis(1));
        let server = serve(onnx::unlinked_sequence_model(), config).await;

        let (a, b) = tokio::join!(
            server.infer(request(array![1.0f32, 2.0])),
            server.infer(request(array![3.0f32, 4.0, 5.0])),
        );

        assert_eq!(a.unwrap().values::<f32>("y"), [2.0, 4.0, 0.0]);
        assert_eq!(b.unwrap().values::<f32>("y"), [6.0, 8.0, 10.0]);
    }

    #[tokio::test]
    async fn invalid_crop_axes_fail_startup() {
        let cases = [
            (
                OutputSpec::new("y").with_crop_axis(0, "x", 1),
                "axis 0 is the batch axis",
            ),
            (
                OutputSpec::new("y").with_crop_axis(1, "z", 1),
                "'z' is not a batched model input",
            ),
            (
                OutputSpec::new("y").with_crop_axis(1, "x", 0),
                "input axis 0 is the batch axis",
            ),
            (
                OutputSpec::new("y").with_crop_axis(1, "x", 5),
                "input 'x' has no axis 5",
            ),
        ];
        for (spec, expected) in cases {
            let config = single_batch_config(2).with_output(spec);
            let message = startup_error(try_serve(onnx::unlinked_sequence_model(), config).await);
            assert!(message.contains(expected), "{message}");
        }
    }
}
