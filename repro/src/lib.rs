#[cfg(test)]
mod tests {
  use std::path::Path;

  use whisper_rs::FullParams;
  use whisper_rs::SamplingStrategy;
  use whisper_rs::WhisperContext;
  use whisper_rs::WhisperContextParameters;


  const SAMPLES: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/samples_jfk.1-16000-i16.raw"
  ));


  fn transmute_bytes<T>(bytes: &[u8]) -> &[T] {
    let (_front, data, _back) = unsafe { bytes.align_to::<T>() };
    debug_assert_eq!(_front, &[]);
    debug_assert_eq!(_back, &[]);
    data
  }


  fn transcribe() {
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("ggml-tiny.en.bin");
    let context = WhisperContext::new_with_params(
      model.to_str().unwrap(),
      WhisperContextParameters::default(),
    ).unwrap();
    let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    let mut state = context.create_state().unwrap();
    let _zero = state
      .full(params.clone(), transmute_bytes(SAMPLES))
      .unwrap();
  }


  #[test]
  fn it_works() {
    transcribe()
  }

  #[test]
  fn oops_no_it_doesnt() {
    transcribe()
  }
}
