use ndarray::{Array3, ArrayD, ArrayViewD, Axis};
use ort_superserve::{Input, Output};

struct MockInput {
    data: Array3<f32>,
}

impl Input for MockInput {
    type Preprocessed = Array3<f32>;

    async fn preprocess(self) -> anyhow::Result<Self::Preprocessed> {
        Ok(self.data)
    }

    fn batch(items: Vec<Self::Preprocessed>) -> anyhow::Result<ArrayD<f32>> {
        let views: Vec<_> = items.iter().map(|a| a.view()).collect();
        let batched = ndarray::stack(Axis(0), &views)?;
        Ok(batched.into_dyn())
    }
}

#[derive(Debug)]
struct MockOutput {
    value: f32,
}

impl Output for MockOutput {
    async fn postprocess(raw: ArrayViewD<'_, f32>) -> anyhow::Result<Self> {
        let first = raw.first().copied().unwrap_or(0.0);
        Ok(MockOutput { value: first })
    }
}

mod config_tests {
    use ort_superserve::ServerConfig;
    use std::time::Duration;

    #[test]
    fn test_default_config() {
        let config = ServerConfig::default();

        assert_eq!(config.num_sessions, 1);
        assert_eq!(config.threads_per_session, 1);
        assert_eq!(config.max_batch_size, 8);
        assert_eq!(config.min_batch_size, 1);
        assert_eq!(config.max_wait_time, Duration::from_millis(10));
    }

    #[test]
    fn test_config_builder() {
        let config = ServerConfig::new()
            .with_num_sessions(4)
            .with_threads_per_session(2)
            .with_max_batch_size(16)
            .with_min_batch_size(2)
            .with_max_wait_time(Duration::from_millis(20));

        assert_eq!(config.num_sessions, 4);
        assert_eq!(config.threads_per_session, 2);
        assert_eq!(config.max_batch_size, 16);
        assert_eq!(config.min_batch_size, 2);
        assert_eq!(config.max_wait_time, Duration::from_millis(20));
    }

    #[test]
    fn test_config_clone() {
        let config = ServerConfig::new()
            .with_num_sessions(2)
            .with_max_batch_size(32);

        let cloned = config.clone();

        assert_eq!(config.num_sessions, cloned.num_sessions);
        assert_eq!(config.max_batch_size, cloned.max_batch_size);
    }
}

mod input_tests {
    use super::*;

    #[tokio::test]
    async fn test_input_preprocess() {
        let data = Array3::<f32>::zeros((3, 224, 224));
        let input = MockInput { data: data.clone() };

        let result = input.preprocess().await.unwrap();

        assert_eq!(result.shape(), [3, 224, 224]);
    }

    #[test]
    fn test_input_batch() {
        let items: Vec<Array3<f32>> = vec![
            Array3::<f32>::zeros((3, 224, 224)),
            Array3::<f32>::ones((3, 224, 224)),
        ];

        let batched = MockInput::batch(items).unwrap();

        assert_eq!(batched.shape(), &[2, 3, 224, 224]);
    }

    #[test]
    fn test_input_batch_empty() {
        let items: Vec<Array3<f32>> = vec![];

        let result = MockInput::batch(items);

        assert!(result.is_err() || result.unwrap().shape().is_empty());
    }
}

mod output_tests {
    use super::*;

    #[tokio::test]
    async fn test_output_postprocess() {
        let data = ndarray::arr1(&[1.0, 2.0, 3.0]);
        let view = data.view().into_dyn();

        let output = MockOutput::postprocess(view).await.unwrap();

        assert_eq!(output.value, 1.0);
    }

    #[tokio::test]
    async fn test_output_postprocess_from_shape() {
        let data =
            ndarray::Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let view = data.view().into_dyn();

        let output = MockOutput::postprocess(view).await.unwrap();

        assert_eq!(output.value, 1.0);
    }
}

mod batch_tests {
    use super::*;

    #[test]
    fn test_batch_single_item() {
        let items: Vec<Array3<f32>> = vec![Array3::<f32>::ones((1, 10, 10))];

        let batched = MockInput::batch(items).unwrap();

        assert_eq!(batched.shape(), &[1, 1, 10, 10]);
    }

    #[test]
    fn test_batch_multiple_items() {
        let items: Vec<Array3<f32>> = vec![
            Array3::<f32>::zeros((3, 32, 32)),
            Array3::<f32>::zeros((3, 32, 32)),
            Array3::<f32>::zeros((3, 32, 32)),
        ];

        let batched = MockInput::batch(items).unwrap();

        assert_eq!(batched.shape(), &[3, 3, 32, 32]);
    }

    #[test]
    fn test_batch_different_shapes_fails() {
        let items: Vec<Array3<f32>> = vec![
            Array3::<f32>::zeros((3, 32, 32)),
            Array3::<f32>::zeros((3, 64, 64)),
        ];

        let result = MockInput::batch(items);

        assert!(result.is_err());
    }
}
