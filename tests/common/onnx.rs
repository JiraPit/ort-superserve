//! Minimal hand-rolled ONNX (protobuf) encoder for building test models.
//!
//! Only the handful of fields needed by the fixture models are supported.
//! Field numbers follow `onnx.proto`.

const ELEM_FLOAT: i64 = 1;
const ELEM_INT64: i64 = 7;
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
}

/// `ValueInfoProto` for a tensor.
fn value_info(name: &str, elem_type: i64, dims: &[Dim]) -> Vec<u8> {
    let mut shape = Vec::new();
    for dim in dims {
        let mut d = Vec::new();
        match dim {
            Dim::Batch => bytes_field(&mut d, 2, b"N"),
            Dim::Fixed(v) => int_field(&mut d, 1, *v),
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

/// Scalar float `TensorProto` used as an initializer.
fn scalar_initializer(name: &str, value: f32) -> Vec<u8> {
    let mut tensor = Vec::new();
    int_field(&mut tensor, 2, ELEM_FLOAT);
    bytes_field(&mut tensor, 8, name.as_bytes());
    bytes_field(&mut tensor, 9, &value.to_le_bytes());
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

pub fn int64_output_model() -> Vec<u8> {
    model(
        &[node("Cast", &["x"], &["y"], &[Attr::Int("to", ELEM_INT64)])],
        &[],
        &[batched_float("x")],
        &[value_info("y", ELEM_INT64, &[Dim::Batch, Dim::Fixed(4)])],
    )
}
