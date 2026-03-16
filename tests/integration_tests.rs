use ndarray::{Array3, ArrayD};
use ort_superserve::{Input, Server, ServerConfig, helpers::batch_array};
use std::path::PathBuf;

struct MockInput {
    data: Array3<f32>,
}

impl Input for MockInput {
    type Preprocessed = Array3<f32>;

    async fn preprocess(self) -> anyhow::Result<Self::Preprocessed> {
        Ok(self.data)
    }

    fn batch(items: Vec<Self::Preprocessed>) -> anyhow::Result<ArrayD<f32>> {
        batch_array(&items)
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
        assert!(!config.preprocess_on_infer);
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

    #[test]
    fn test_config_preprocess_on_infer() {
        let config_false = ServerConfig::new()
            .with_preprocess_on_infer(false);
        assert!(!config_false.preprocess_on_infer);

        let config_true = ServerConfig::new()
            .with_preprocess_on_infer(true);
        assert!(config_true.preprocess_on_infer);
    }

    #[test]
    fn test_config_with_input_output_names() {
        let config = ServerConfig::new()
            .with_input_name("input_tensor")
            .with_output_name("output_tensor");

        assert_eq!(config.input_name, Some("input_tensor".to_string()));
        assert_eq!(config.output_name, Some("output_tensor".to_string()));
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

    #[test]
    fn test_batch_empty_fails() {
        let items: Vec<Array3<f32>> = vec![];
        let result = MockInput::batch(items);
        assert!(result.is_err());
    }
}

fn get_test_model_path() -> PathBuf {
    PathBuf::from("test_assets/mobilenetv2-12-int8/mobilenetv2-12-int8.onnx")
}

fn get_test_image_path() -> PathBuf {
    PathBuf::from("test_assets/mobilenetv2-12-int8/images/0.png")
}

fn check_model_exists() -> bool {
    get_test_model_path().exists()
}

fn check_image_exists() -> bool {
    get_test_image_path().exists()
}

fn panic_with_download_instructions() -> ! {
    panic!(
        "Test model not found. Please download it first:\n\
         \n\
         From the project root, run:\n\
         cd download_test_assets && uv run download_test_assets.py --model mobilenet\n\
         \n\
         This will download the MobileNetV2 model and test images to test_assets/mobilenetv2-12-int8/"
    )
}

mod server_tests {
    use super::*;
    use ort::session::builder::GraphOptimizationLevel;
    use ort_superserve::Output;
    use ndarray::ArrayViewD;
    use std::time::Duration;
    use tokio::test;

    fn load_png_image(path: &PathBuf) -> Vec<u8> {
        std::fs::read(path).expect("Failed to read PNG image")
    }

    fn create_input_array(image_bytes: Vec<u8>) -> Array3<f32> {
        let decoder = png::Decoder::new(std::io::Cursor::new(image_bytes));
        let mut reader = decoder.read_info().expect("Failed to read PNG info");
        let mut buf = vec![0; reader.output_buffer_size()];
        reader.next_frame(&mut buf).expect("Failed to decode PNG");
        
        let info = reader.info();
        let width = info.width as usize;
        let height = info.height as usize;
        
        let mut array = Array3::<f32>::zeros((3, height, width));
        
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                let r = buf[idx] as f32 / 255.0;
                let g = buf[idx + 1] as f32 / 255.0;
                let b = buf[idx + 2] as f32 / 255.0;
                
                array[[0, y, x]] = (r - 0.485) / 0.229;
                array[[1, y, x]] = (g - 0.456) / 0.224;
                array[[2, y, x]] = (b - 0.406) / 0.225;
            }
        }
        
        array
    }

    struct ImageInput {
        data: Array3<f32>,
    }

    impl Input for ImageInput {
        type Preprocessed = Array3<f32>;

        async fn preprocess(self) -> anyhow::Result<Self::Preprocessed> {
            Ok(self.data)
        }

        fn batch(items: Vec<Self::Preprocessed>) -> anyhow::Result<ArrayD<f32>> {
            batch_array(&items)
        }
    }

    struct ImageOutput {
        logits: Vec<f32>,
    }

    impl Output for ImageOutput {
        async fn postprocess(raw: ArrayViewD<'_, f32>) -> anyhow::Result<Self> {
            Ok(ImageOutput {
                logits: raw.iter().copied().collect(),
            })
        }
    }

    #[test]
    async fn test_server_preprocess_on_infer_false() {
        if !check_model_exists() || !check_image_exists() {
            panic_with_download_instructions();
        }

        let config = ServerConfig::new()
            .with_num_sessions(1)
            .with_threads_per_session(1)
            .with_max_batch_size(8)
            .with_min_batch_size(1)
            .with_max_wait_time(Duration::from_millis(10))
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .with_preprocess_on_infer(false);

        let server = Server::<ImageInput, ImageOutput>::from_file(&get_test_model_path(), config)
            .await
            .expect("Failed to create server");

        let image_bytes = load_png_image(&get_test_image_path());
        let input_array = create_input_array(image_bytes);
        let input = ImageInput { data: input_array };

        let output = server.infer(input).await.expect("Inference failed");
        
        assert!(!output.logits.is_empty(), "Output logits should not be empty");
        
        server.shutdown();
    }

    #[test]
    async fn test_server_preprocess_on_infer_true() {
        if !check_model_exists() || !check_image_exists() {
            panic_with_download_instructions();
        }

        let config = ServerConfig::new()
            .with_num_sessions(1)
            .with_threads_per_session(1)
            .with_max_batch_size(8)
            .with_min_batch_size(1)
            .with_max_wait_time(Duration::from_millis(10))
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .with_preprocess_on_infer(true);

        let server = Server::<ImageInput, ImageOutput>::from_file(&get_test_model_path(), config)
            .await
            .expect("Failed to create server");

        let image_bytes = load_png_image(&get_test_image_path());
        let input_array = create_input_array(image_bytes);
        let input = ImageInput { data: input_array };

        let output = server.infer(input).await.expect("Inference failed");
        
        assert!(!output.logits.is_empty(), "Output logits should not be empty");
        
        server.shutdown();
    }

    #[test]
    async fn test_server_with_explicit_tensor_names() {
        if !check_model_exists() || !check_image_exists() {
            panic_with_download_instructions();
        }

        let config = ServerConfig::new()
            .with_num_sessions(1)
            .with_threads_per_session(1)
            .with_max_batch_size(8)
            .with_min_batch_size(1)
            .with_max_wait_time(Duration::from_millis(10))
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .with_input_name("input")
            .with_output_name("output");

        let server = Server::<ImageInput, ImageOutput>::from_file(&get_test_model_path(), config)
            .await
            .expect("Failed to create server");

        let image_bytes = load_png_image(&get_test_image_path());
        let input_array = create_input_array(image_bytes);
        let input = ImageInput { data: input_array };

        let output = server.infer(input).await.expect("Inference failed");
        
        assert!(!output.logits.is_empty(), "Output logits should not be empty");
        
        server.shutdown();
    }

    #[test]
    async fn test_server_auto_detect_tensor_names() {
        if !check_model_exists() || !check_image_exists() {
            panic_with_download_instructions();
        }

        let config = ServerConfig::new()
            .with_num_sessions(1)
            .with_threads_per_session(1)
            .with_max_batch_size(8)
            .with_min_batch_size(1)
            .with_max_wait_time(Duration::from_millis(10))
            .with_optimization_level(GraphOptimizationLevel::Level3);

        let server = Server::<ImageInput, ImageOutput>::from_file(&get_test_model_path(), config)
            .await
            .expect("Failed to create server");

        let image_bytes = load_png_image(&get_test_image_path());
        let input_array = create_input_array(image_bytes);
        let input = ImageInput { data: input_array };

        let output = server.infer(input).await.expect("Inference failed");
        
        assert!(!output.logits.is_empty(), "Output logits should not be empty");
        
        server.shutdown();
    }

    #[test]
    async fn test_server_multiple_inferences() {
        if !check_model_exists() || !check_image_exists() {
            panic_with_download_instructions();
        }

        let config = ServerConfig::new()
            .with_num_sessions(1)
            .with_threads_per_session(1)
            .with_max_batch_size(8)
            .with_min_batch_size(1)
            .with_max_wait_time(Duration::from_millis(10))
            .with_optimization_level(GraphOptimizationLevel::Level3);

        let server = Server::<ImageInput, ImageOutput>::from_file(&get_test_model_path(), config)
            .await
            .expect("Failed to create server");

        let image_bytes = load_png_image(&get_test_image_path());
        
        // Run multiple sequential inferences
        for _ in 0..5 {
            let input_array = create_input_array(image_bytes.clone());
            let input = ImageInput { data: input_array };
            let output = server.clone().infer(input).await.expect("Inference failed");
            assert!(!output.logits.is_empty());
        }
        
        server.shutdown();
    }

    #[test]
    async fn test_server_clone_and_concurrent_inference() {
        if !check_model_exists() || !check_image_exists() {
            panic_with_download_instructions();
        }

        let config = ServerConfig::new()
            .with_num_sessions(1)
            .with_threads_per_session(1)
            .with_max_batch_size(8)
            .with_min_batch_size(1)
            .with_max_wait_time(Duration::from_millis(10))
            .with_optimization_level(GraphOptimizationLevel::Level3);

        let server = Server::<ImageInput, ImageOutput>::from_file(&get_test_model_path(), config)
            .await
            .expect("Failed to create server");

        let image_bytes = load_png_image(&get_test_image_path());
        
        let handles: Vec<_> = (0..3)
            .map(|_| {
                let server = server.clone();
                let image_bytes = image_bytes.clone();
                tokio::spawn(async move {
                    let input_array = create_input_array(image_bytes);
                    let input = ImageInput { data: input_array };
                    server.infer(input).await.expect("Inference failed")
                })
            })
            .collect();

        for handle in handles {
            let output = handle.await.expect("Task failed");
            assert!(!output.logits.is_empty());
        }
        
        server.shutdown();
    }
}