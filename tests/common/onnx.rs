//! Minimal hand-rolled ONNX (protobuf) encoder for building test models.
//!
//! Only the handful of fields needed by the fixture models are supported.
//! Field numbers follow `onnx.proto`.

const ELEM_FLOAT: i64 = 1;
const ELEM_UINT8: i64 = 2;
const ELEM_INT64: i64 = 7;
const ELEM_BOOL: i64 = 9;
const ELEM_DOUBLE: i64 = 11;
const ATTR_INT: i64 = 2;
const ATTR_INTS: i64 = 7;

fn varint(buf: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        buf.push((value as u8) | 0x80);
        value >>= 7;
    }
    buf.push(value as u8);
}

fn int_field(buf: &mut Vec<u8>, field: u64, value: i64) {
    varint(buf, field << 3);
    varint(buf, value as u64);
}

fn bytes_field(buf: &mut Vec<u8>, field: u64, data: &[u8]) {
    varint(buf, (field << 3) | 2);
    varint(buf, data.len() as u64);
    buf.extend_from_slice(data);
}

enum Dim {
    Batch,
    Fixed(i64),
    /// A dynamic dimension with this symbolic name.
    Named(&'static str),
}

/// `ValueInfoProto` for a tensor.
fn value_info(name: &str, elem_type: i64, dims: &[Dim]) -> Vec<u8> {
    let mut shape = Vec::new();
    for dim in dims {
        let mut d = Vec::new();
        match dim {
            Dim::Batch => bytes_field(&mut d, 2, b"N"),
            Dim::Fixed(v) => int_field(&mut d, 1, *v),
            Dim::Named(name) => bytes_field(&mut d, 2, name.as_bytes()),
        }
        bytes_field(&mut shape, 1, &d);
    }

    let mut tensor_type = Vec::new();
    int_field(&mut tensor_type, 1, elem_type);
    bytes_field(&mut tensor_type, 2, &shape);

    let mut type_proto = Vec::new();
    bytes_field(&mut type_proto, 1, &tensor_type);

    let mut info = Vec::new();
    bytes_field(&mut info, 1, name.as_bytes());
    bytes_field(&mut info, 2, &type_proto);
    info
}

/// `ValueInfoProto` for a tensor whose shape is left unspecified.
fn unshaped_value_info(name: &str, elem_type: i64) -> Vec<u8> {
    let mut tensor_type = Vec::new();
    int_field(&mut tensor_type, 1, elem_type);

    let mut type_proto = Vec::new();
    bytes_field(&mut type_proto, 1, &tensor_type);

    let mut info = Vec::new();
    bytes_field(&mut info, 1, name.as_bytes());
    bytes_field(&mut info, 2, &type_proto);
    info
}

/// Scalar float `TensorProto` used as an initializer.
fn scalar_initializer(name: &str, value: f32) -> Vec<u8> {
    let mut tensor = Vec::new();
    int_field(&mut tensor, 2, ELEM_FLOAT);
    bytes_field(&mut tensor, 8, name.as_bytes());
    bytes_field(&mut tensor, 9, &value.to_le_bytes());
    tensor
}

/// `TensorProto` initializer with a shape, holding `f32` values.
fn float_initializer(name: &str, dims: &[i64], values: &[f32]) -> Vec<u8> {
    let mut tensor = Vec::new();
    for d in dims {
        int_field(&mut tensor, 1, *d);
    }
    int_field(&mut tensor, 2, ELEM_FLOAT);
    bytes_field(&mut tensor, 8, name.as_bytes());
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    bytes_field(&mut tensor, 9, &raw);
    tensor
}

/// `TensorProto` initializer with a shape (empty for a scalar), holding `i64` values.
fn int64_initializer(name: &str, dims: &[i64], values: &[i64]) -> Vec<u8> {
    let mut tensor = Vec::new();
    for d in dims {
        int_field(&mut tensor, 1, *d);
    }
    int_field(&mut tensor, 2, ELEM_INT64);
    bytes_field(&mut tensor, 8, name.as_bytes());
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    bytes_field(&mut tensor, 9, &raw);
    tensor
}

enum Attr<'a> {
    Int(&'a str, i64),
    Ints(&'a str, &'a [i64]),
}

fn node(op_type: &str, inputs: &[&str], outputs: &[&str], attrs: &[Attr]) -> Vec<u8> {
    let mut n = Vec::new();
    for input in inputs {
        bytes_field(&mut n, 1, input.as_bytes());
    }
    for output in outputs {
        bytes_field(&mut n, 2, output.as_bytes());
    }
    bytes_field(&mut n, 4, op_type.as_bytes());
    for attr in attrs {
        let mut a = Vec::new();
        match attr {
            Attr::Int(name, v) => {
                bytes_field(&mut a, 1, name.as_bytes());
                int_field(&mut a, 3, *v);
                int_field(&mut a, 20, ATTR_INT);
            }
            Attr::Ints(name, vs) => {
                bytes_field(&mut a, 1, name.as_bytes());
                for v in vs.iter() {
                    int_field(&mut a, 8, *v);
                }
                int_field(&mut a, 20, ATTR_INTS);
            }
        }
        bytes_field(&mut n, 5, &a);
    }
    n
}

/// `ModelProto` (IR v7, opset 13) with a single graph.
fn model(
    nodes: &[Vec<u8>],
    initializers: &[Vec<u8>],
    inputs: &[Vec<u8>],
    outputs: &[Vec<u8>],
) -> Vec<u8> {
    let mut graph = Vec::new();
    for n in nodes {
        bytes_field(&mut graph, 1, n);
    }
    bytes_field(&mut graph, 2, b"test_graph");
    for init in initializers {
        bytes_field(&mut graph, 5, init);
    }
    for input in inputs {
        bytes_field(&mut graph, 11, input);
    }
    for output in outputs {
        bytes_field(&mut graph, 12, output);
    }

    let mut opset = Vec::new();
    bytes_field(&mut opset, 1, b"");
    int_field(&mut opset, 2, 13);

    let mut m = Vec::new();
    int_field(&mut m, 1, 7);
    bytes_field(&mut m, 7, &graph);
    bytes_field(&mut m, 8, &opset);
    m
}

fn batched_float(name: &str) -> Vec<u8> {
    value_info(name, ELEM_FLOAT, &[Dim::Batch, Dim::Fixed(4)])
}

pub fn doubling_model() -> Vec<u8> {
    model(
        &[node("Mul", &["x", "two"], &["y"], &[])],
        &[scalar_initializer("two", 2.0)],
        &[batched_float("x")],
        &[batched_float("y")],
    )
}

pub fn two_outputs_model() -> Vec<u8> {
    model(
        &[
            node("Mul", &["x", "two"], &["doubled"], &[]),
            node("Add", &["x", "one"], &["incremented"], &[]),
        ],
        &[
            scalar_initializer("two", 2.0),
            scalar_initializer("one", 1.0),
        ],
        &[batched_float("x")],
        &[batched_float("doubled"), batched_float("incremented")],
    )
}

pub fn batch_collapsing_model() -> Vec<u8> {
    model(
        &[node(
            "ReduceMean",
            &["x"],
            &["y"],
            &[Attr::Ints("axes", &[0]), Attr::Int("keepdims", 1)],
        )],
        &[],
        &[batched_float("x")],
        &[value_info("y", ELEM_FLOAT, &[Dim::Fixed(1), Dim::Fixed(4)])],
    )
}

pub fn squeezing_model() -> Vec<u8> {
    model(
        &[node("Squeeze", &["x"], &["y"], &[])],
        &[],
        &[value_info("x", ELEM_FLOAT, &[Dim::Batch, Dim::Fixed(1)])],
        &[unshaped_value_info("y", ELEM_FLOAT)],
    )
}

pub fn int64_output_model() -> Vec<u8> {
    model(
        &[node("Cast", &["x"], &["y"], &[Attr::Int("to", ELEM_INT64)])],
        &[],
        &[batched_float("x")],
        &[value_info("y", ELEM_INT64, &[Dim::Batch, Dim::Fixed(4)])],
    )
}

fn tokens(name: &str, batch: Dim) -> Vec<u8> {
    value_info(name, ELEM_INT64, &[batch, Dim::Named("sequence")])
}

/// Token model with inputs `input_ids` and `attention_mask` (`i64`, `[batch,
/// sequence]`) and outputs:
/// - `masked = ids * mask` (`f32`, `[batch, sequence]`),
/// - `labels = ids` (`i64`, `[batch, sequence]`),
/// - `max_id`, the largest id including padding (`f32`, `[batch]`).
fn tokens_model_with_batch(batch: fn() -> Dim) -> Vec<u8> {
    model(
        &[
            node(
                "Cast",
                &["input_ids"],
                &["ids_f"],
                &[Attr::Int("to", ELEM_FLOAT)],
            ),
            node(
                "Cast",
                &["attention_mask"],
                &["mask_f"],
                &[Attr::Int("to", ELEM_FLOAT)],
            ),
            node("Mul", &["ids_f", "mask_f"], &["masked"], &[]),
            node("Identity", &["input_ids"], &["labels"], &[]),
            node(
                "ReduceMax",
                &["ids_f"],
                &["max_id"],
                &[Attr::Ints("axes", &[1]), Attr::Int("keepdims", 0)],
            ),
        ],
        &[],
        &[
            tokens("input_ids", batch()),
            tokens("attention_mask", batch()),
        ],
        &[
            value_info("masked", ELEM_FLOAT, &[batch(), Dim::Named("sequence")]),
            value_info("labels", ELEM_INT64, &[batch(), Dim::Named("sequence")]),
            value_info("max_id", ELEM_FLOAT, &[batch()]),
        ],
    )
}

pub fn tokens_model() -> Vec<u8> {
    tokens_model_with_batch(|| Dim::Batch)
}

/// [`tokens_model`] exported with a fixed batch size of 4.
pub fn fixed_batch_tokens_model() -> Vec<u8> {
    tokens_model_with_batch(|| Dim::Fixed(4))
}

/// `u8` input `x: [N, 3]` with outputs `as_f64` (`f64`) and `nonzero` (`bool`).
pub fn element_types_model() -> Vec<u8> {
    model(
        &[
            node("Cast", &["x"], &["as_f64"], &[Attr::Int("to", ELEM_DOUBLE)]),
            node("Cast", &["x"], &["nonzero"], &[Attr::Int("to", ELEM_BOOL)]),
        ],
        &[],
        &[value_info("x", ELEM_UINT8, &[Dim::Batch, Dim::Fixed(3)])],
        &[
            value_info("as_f64", ELEM_DOUBLE, &[Dim::Batch, Dim::Fixed(3)]),
            value_info("nonzero", ELEM_BOOL, &[Dim::Batch, Dim::Fixed(3)]),
        ],
    )
}

/// `y = x * scale` with `x: [N, 4]` and a scalar `scale` input shared by the batch.
pub fn scaled_model() -> Vec<u8> {
    model(
        &[node("Mul", &["x", "scale"], &["y"], &[])],
        &[],
        &[batched_float("x"), value_info("scale", ELEM_FLOAT, &[])],
        &[batched_float("y")],
    )
}

/// NMS-like model for `x: [N, 2]`. Every batch entry `i` produces two rows: row
/// `i` of a first block in batch order, then row `2N - 1 - i` of a second block
/// in reverse batch order, so rows of one entry are not evenly spaced:
/// - `detections` (`[2N, 3]`): `[i, x0, x1]` then `[i, x0 + 10, x1 + 10]`,
/// - `boxes` (`[2N, 2]`): the same rows without the batch index column,
/// - `batch_index` (`i64`, `[2N]`): the batch index of each `boxes` row.
///
/// Outputs are declared without a shape, like real NMS outputs.
pub fn nms_model() -> Vec<u8> {
    model(
        &[
            node("Shape", &["x"], &["shape"], &[]),
            node(
                "Gather",
                &["shape", "zero"],
                &["n"],
                &[Attr::Int("axis", 0)],
            ),
            node("Range", &["zero", "n", "one"], &["index"], &[]),
            node(
                "Cast",
                &["index"],
                &["index_f"],
                &[Attr::Int("to", ELEM_FLOAT)],
            ),
            node("Unsqueeze", &["index_f", "axis_1"], &["index_col"], &[]),
            node("Add", &["x", "ten"], &["x10"], &[]),
            node(
                "Concat",
                &["index_col", "x"],
                &["first"],
                &[Attr::Int("axis", 1)],
            ),
            node(
                "Concat",
                &["index_col", "x10"],
                &["second"],
                &[Attr::Int("axis", 1)],
            ),
            node(
                "Slice",
                &["second", "last", "before_first", "axis_0", "minus_one"],
                &["second_rev"],
                &[],
            ),
            node(
                "Slice",
                &["x10", "last", "before_first", "axis_0", "minus_one"],
                &["x10_rev"],
                &[],
            ),
            node(
                "Slice",
                &["index", "last", "before_first", "axis_0", "minus_one"],
                &["index_rev"],
                &[],
            ),
            node(
                "Concat",
                &["first", "second_rev"],
                &["detections"],
                &[Attr::Int("axis", 0)],
            ),
            node(
                "Concat",
                &["x", "x10_rev"],
                &["boxes"],
                &[Attr::Int("axis", 0)],
            ),
            node(
                "Concat",
                &["index", "index_rev"],
                &["batch_index"],
                &[Attr::Int("axis", 0)],
            ),
        ],
        &[
            int64_initializer("zero", &[], &[0]),
            int64_initializer("one", &[], &[1]),
            int64_initializer("axis_0", &[1], &[0]),
            int64_initializer("axis_1", &[1], &[1]),
            // Slice parameters that reverse axis 0.
            int64_initializer("last", &[1], &[-1]),
            int64_initializer("before_first", &[1], &[i64::MIN]),
            int64_initializer("minus_one", &[1], &[-1]),
            scalar_initializer("ten", 10.0),
        ],
        &[value_info("x", ELEM_FLOAT, &[Dim::Batch, Dim::Fixed(2)])],
        &[
            unshaped_value_info("detections", ELEM_FLOAT),
            unshaped_value_info("boxes", ELEM_FLOAT),
            unshaped_value_info("batch_index", ELEM_INT64),
        ],
    )
}

/// NMS-like model exported with a fixed batch size of 4: `x: [4, 2]`, output
/// `detections = [i, x0, x1]` for every batch slot `i`, padded slots included.
pub fn fixed_batch_nms_model() -> Vec<u8> {
    model(
        &[node(
            "Concat",
            &["index", "x"],
            &["detections"],
            &[Attr::Int("axis", 1)],
        )],
        &[float_initializer("index", &[4, 1], &[0.0, 1.0, 2.0, 3.0])],
        &[value_info("x", ELEM_FLOAT, &[Dim::Fixed(4), Dim::Fixed(2)])],
        &[unshaped_value_info("detections", ELEM_FLOAT)],
    )
}

/// `y = x * 2` exported with a fixed batch size: `x, y: [batch_size, 4]`.
pub fn fixed_batch_doubling_model(batch_size: i64) -> Vec<u8> {
    let float = |name| value_info(name, ELEM_FLOAT, &[Dim::Fixed(batch_size), Dim::Fixed(4)]);
    model(
        &[node("Mul", &["x", "two"], &["y"], &[])],
        &[scalar_initializer("two", 2.0)],
        &[float("x")],
        &[float("y")],
    )
}

/// Fixed batch size of 4: `x: [4, 4]`, output `y: [1, 4]`, the per-column
/// maximum over all four rows, padded rows included.
pub fn fixed_batch_column_max_model() -> Vec<u8> {
    model(
        &[node(
            "ReduceMax",
            &["x"],
            &["y"],
            &[Attr::Ints("axes", &[0]), Attr::Int("keepdims", 1)],
        )],
        &[],
        &[value_info("x", ELEM_FLOAT, &[Dim::Fixed(4), Dim::Fixed(4)])],
        &[value_info("y", ELEM_FLOAT, &[Dim::Fixed(1), Dim::Fixed(4)])],
    )
}

/// Fixed batch size of 4 with a data-dependent output: `y` (`i64`,
/// `[nonzero, 2]`) lists the `[row, column]` of every non-zero element of `x: [4, 4]`.
pub fn fixed_batch_nonzero_model() -> Vec<u8> {
    model(
        &[
            node("NonZero", &["x"], &["positions"], &[]),
            node("Transpose", &["positions"], &["y"], &[]),
        ],
        &[],
        &[value_info("x", ELEM_FLOAT, &[Dim::Fixed(4), Dim::Fixed(4)])],
        &[value_info(
            "y",
            ELEM_INT64,
            &[Dim::Named("nonzero"), Dim::Fixed(2)],
        )],
    )
}

/// `y = x` where `x` is `[N, 4]` but `y` declares axis 0 as another dimension, `M`.
pub fn renamed_batch_axis_model() -> Vec<u8> {
    model(
        &[node("Identity", &["x"], &["y"], &[])],
        &[],
        &[batched_float("x")],
        &[value_info(
            "y",
            ELEM_FLOAT,
            &[Dim::Named("M"), Dim::Fixed(4)],
        )],
    )
}

/// `y = x * 2` for `x: [N, sequence]`, with `y` declared `[N, unk__0]`, an
/// exporter-generated name that does not link it to `sequence`.
pub fn unlinked_sequence_model() -> Vec<u8> {
    model(
        &[node("Mul", &["x", "two"], &["y"], &[])],
        &[scalar_initializer("two", 2.0)],
        &[value_info(
            "x",
            ELEM_FLOAT,
            &[Dim::Batch, Dim::Named("sequence")],
        )],
        &[value_info(
            "y",
            ELEM_FLOAT,
            &[Dim::Batch, Dim::Named("unk__0")],
        )],
    )
}

/// `y = x * 2` for `x: [N, "sequence"]`, with no shape declared for `y`.
///
/// `y` is reshaped to the shape of `x` filtered through `Compress`, whose
/// output length is data-dependent, so ONNX Runtime cannot infer even the
/// rank of `y`.
pub fn unshaped_sequence_output_model() -> Vec<u8> {
    model(
        &[
            node("Mul", &["x", "two"], &["doubled"], &[]),
            node("Shape", &["x"], &["shape"], &[]),
            node("Greater", &["shape", "zero"], &["keep"], &[]),
            node("Compress", &["shape", "keep"], &["kept_shape"], &[]),
            node("Reshape", &["doubled", "kept_shape"], &["y"], &[]),
        ],
        &[
            scalar_initializer("two", 2.0),
            int64_initializer("zero", &[], &[0]),
        ],
        &[value_info(
            "x",
            ELEM_FLOAT,
            &[Dim::Batch, Dim::Named("sequence")],
        )],
        &[unshaped_value_info("y", ELEM_FLOAT)],
    )
}

/// Inputs `a: [2, 4]` and `b: [3, 4]` with different fixed batch sizes.
pub fn conflicting_fixed_batch_model() -> Vec<u8> {
    model(
        &[
            node("Identity", &["a"], &["a_out"], &[]),
            node("Identity", &["b"], &["b_out"], &[]),
        ],
        &[],
        &[
            value_info("a", ELEM_FLOAT, &[Dim::Fixed(2), Dim::Fixed(4)]),
            value_info("b", ELEM_FLOAT, &[Dim::Fixed(3), Dim::Fixed(4)]),
        ],
        &[
            value_info("a_out", ELEM_FLOAT, &[Dim::Fixed(2), Dim::Fixed(4)]),
            value_info("b_out", ELEM_FLOAT, &[Dim::Fixed(3), Dim::Fixed(4)]),
        ],
    )
}
