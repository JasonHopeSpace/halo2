//! This module provides "ZK Acceleration Layer" traits
//! to abstract away the execution engine for performance-critical primitives.
//!
//! Terminology
//! -----------
//!
//! We use the name Backend+Engine for concrete implementations of ZalEngine.
//! For example H2cEngine for pure Halo2curves implementation.
//!
//! Alternative names considered were Executor or Driver however
//! - executor is already used in Rust (and the name is long)
//! - driver will be confusing as we work quite low-level with GPUs and FPGAs.
//!
//! Unfortunately the "Engine" name is used in bn256 for pairings.
//! Fortunately a ZalEngine is only used in the prover (at least for now)
//! while "pairing engine" is only used in the verifier
//!
//! Initialization design space
//! ---------------------------
//!
//! It is recommended that ZAL backends provide:
//! - an initialization function:
//!   - either "fn new() -> ZalEngine" for simple libraries
//!   - or a builder pattern for complex initializations
//! - a shutdown function or document when it is not needed (when it's a global threadpool like Rayon for example).
//!
//! Backends might want to add as an option:
//! - The number of threads (CPU)
//! - The device(s) to run on (multi-sockets machines, multi-GPUs machines, ...)
//! - The curve (JIT-compiled backend)
//!
//! Descriptors
//! ---------------------------
//!
//! Descriptors enable providers to configure opaque details on data
//! when doing repeated computations with the same input(s).
//! For example:
//! - Pointer(s) caching to limit data movement between CPU and GPU, FPGAs
//! - Length of data
//! - data in layout:
//!    - canonical or Montgomery fields, unsaturated representation, endianness
//!    - jacobian or projective coordinates or maybe even Twisted Edwards for faster elliptic curve additions,
//!    - FFT: canonical or bit-reversed permuted
//! - data out layout
//! - Device(s) ID
//!
//! For resources that need special cleanup like GPU memory, a custom `Drop` is required.
//!
//! Note that resources can also be stored in the engine in a hashmap
//! and an integer ID or a pointer can be opaquely given as a descriptor.

// The ZK Accel Layer API
// ---------------------------------------------------
pub mod traits {
    use halo2curves::CurveAffine;

    pub trait MsmAccel<C: CurveAffine> {
        fn msm(&self, coeffs: &[C::Scalar], base: &[C]) -> C::Curve;

        // Caching API
        // -------------------------------------------------
        // From here we propose an extended API
        // that allows reusing coeffs and/or the base points
        //
        // This is inspired by CuDNN API (Nvidia GPU)
        // and oneDNN API (CPU, OpenCL) https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnn-ops-infer-so-opaque
        // usage of descriptors
        //
        // https://github.com/oneapi-src/oneDNN/blob/master/doc/programming_model/basic_concepts.md
        //
        // Descriptors are opaque pointers that hold the input in a format suitable for the accelerator engine.
        // They may be:
        // - Input moved on accelerator device (only once for repeated calls)
        // - Endianess conversion
        // - Converting from Montgomery to Canonical form
        // - Input changed from Projective to Jacobian coordinates or even to a Twisted Edwards curve.
        // - other form of expensive preprocessing
        type CoeffsDescriptor<'c>;
        type BaseDescriptor<'b>;

        fn get_coeffs_descriptor<'c>(&self, coeffs: &'c [C::Scalar]) -> Self::CoeffsDescriptor<'c>;
        fn get_base_descriptor<'b>(&self, base: &'b [C]) -> Self::BaseDescriptor<'b>;

        fn msm_with_cached_scalars(
            &self,
            coeffs: &Self::CoeffsDescriptor<'_>,
            base: &[C],
        ) -> C::Curve;

        fn msm_with_cached_base(
            &self,
            coeffs: &[C::Scalar],
            base: &Self::BaseDescriptor<'_>,
        ) -> C::Curve;

        fn msm_with_cached_inputs(
            &self,
            coeffs: &Self::CoeffsDescriptor<'_>,
            base: &Self::BaseDescriptor<'_>,
        ) -> C::Curve;
        // Execute MSM according to descriptors
        // Unsure of naming, msm_with_cached_inputs, msm_apply, msm_cached, msm_with_descriptors, ...
    }
}

// ZAL using Halo2curves as a backend
// ---------------------------------------------------

pub mod impls {
    use std::marker::PhantomData;

    use crate::zal::traits::MsmAccel;
    use halo2curves::msm::best_multiexp;
    use halo2curves::CurveAffine;

    // Halo2curve Backend
    // ---------------------------------------------------
    #[derive(Default)]
    pub struct H2cEngine;

    pub struct H2cMsmCoeffsDesc<'c, C: CurveAffine> {
        raw: &'c [C::Scalar],
    }

    pub struct H2cMsmBaseDesc<'b, C: CurveAffine> {
        raw: &'b [C],
    }

    impl H2cEngine {
        pub fn new() -> Self {
            Self {}
        }
    }

    impl<C: CurveAffine> MsmAccel<C> for H2cEngine {
        fn msm(&self, coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
            best_multiexp(coeffs, bases)
        }

        // Caching API
        // -------------------------------------------------

        type CoeffsDescriptor<'c> = H2cMsmCoeffsDesc<'c, C>;
        type BaseDescriptor<'b> = H2cMsmBaseDesc<'b, C>;

        fn get_coeffs_descriptor<'c>(&self, coeffs: &'c [C::Scalar]) -> Self::CoeffsDescriptor<'c> {
            // Do expensive device/library specific preprocessing here
            Self::CoeffsDescriptor { raw: coeffs }
        }
        fn get_base_descriptor<'b>(&self, base: &'b [C]) -> Self::BaseDescriptor<'b> {
            Self::BaseDescriptor { raw: base }
        }

        fn msm_with_cached_scalars(
            &self,
            coeffs: &Self::CoeffsDescriptor<'_>,
            base: &[C],
        ) -> C::Curve {
            best_multiexp(coeffs.raw, base)
        }

        fn msm_with_cached_base(
            &self,
            coeffs: &[C::Scalar],
            base: &Self::BaseDescriptor<'_>,
        ) -> C::Curve {
            best_multiexp(coeffs, base.raw)
        }

        fn msm_with_cached_inputs(
            &self,
            coeffs: &Self::CoeffsDescriptor<'_>,
            base: &Self::BaseDescriptor<'_>,
        ) -> C::Curve {
            best_multiexp(coeffs.raw, base.raw)
        }
    }

    // Backend-agnostic engine objects
    // ---------------------------------------------------
    #[derive(Debug)]
    pub struct PlonkEngine<C: CurveAffine, MsmEngine: MsmAccel<C>> {
        pub msm_backend: MsmEngine,
        _marker: PhantomData<C>, // compiler complains about unused C otherwise
    }

    #[derive(Default)]
    pub struct PlonkEngineConfig<C, M> {
        curve: PhantomData<C>,
        msm_backend: M,
    }

    #[derive(Default)]
    pub struct NoCurve;

    #[derive(Default)]
    pub struct HasCurve<C: CurveAffine>(PhantomData<C>);

    #[derive(Default)]
    pub struct NoMsmEngine;

    pub struct HasMsmEngine<C: CurveAffine, M: MsmAccel<C>>(M, PhantomData<C>);

    impl PlonkEngineConfig<NoCurve, NoMsmEngine> {
        pub fn new() -> PlonkEngineConfig<NoCurve, NoMsmEngine> {
            Default::default()
        }

        pub fn set_curve<C: CurveAffine>(self) -> PlonkEngineConfig<HasCurve<C>, NoMsmEngine> {
            Default::default()
        }

        pub fn build_default<C: CurveAffine>() -> PlonkEngine<C, H2cEngine> {
            PlonkEngine {
                msm_backend: H2cEngine::new(),
                _marker: Default::default(),
            }
        }
    }

    impl<C: CurveAffine, M> PlonkEngineConfig<HasCurve<C>, M> {
        pub fn set_msm<MsmEngine: MsmAccel<C>>(
            self,
            engine: MsmEngine,
        ) -> PlonkEngineConfig<HasCurve<C>, HasMsmEngine<C, MsmEngine>> {
            // Copy all other parameters
            let Self { curve, .. } = self;
            // Return with modified MSM engine
            PlonkEngineConfig {
                curve,
                msm_backend: HasMsmEngine(engine, Default::default()),
            }
        }
    }

    impl<C: CurveAffine, M: MsmAccel<C>> PlonkEngineConfig<HasCurve<C>, HasMsmEngine<C, M>> {
        pub fn build(self) -> PlonkEngine<C, M> {
            PlonkEngine {
                msm_backend: self.msm_backend.0,
                _marker: Default::default(),
            }
        }
    }
}

pub mod impls_gpu {

    use ark_std::{end_timer, start_timer};
    use crate::GLOBAL_DEVICE_MANAGER;
    use std::marker::PhantomData;
    use crate::zal::traits::MsmAccel;
    use halo2curves::CurveAffine;

    // GPU-> Panda Backend
    // ---------------------------------------------------
    #[derive(Default)]
    pub struct GPUEngine;

    pub struct GPUMsmCoeffsDesc<'b, C: CurveAffine> {
        raw: usize,
        _marker: PhantomData<&'b C>,
    }
    pub struct GPUMsmBaseDesc<'b, C: CurveAffine> {
        raw: usize,
        _marker: PhantomData<&'b C>,
    }

    impl GPUEngine {
        pub fn new() -> Self {
            Self {}
        }
    }

    impl<C: CurveAffine> MsmAccel<C> for GPUEngine {
        fn msm(&self, coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
            let mut binding = GLOBAL_DEVICE_MANAGER.lock().unwrap();
            let device_manager_handle = binding.get_handle_mut();

            let t1 = start_timer!(|| format!("execute_msm"));
            let mut result_datas = device_manager_handle.execute_msm::<C>(coeffs, bases).unwrap();
            end_timer!(t1);

            let result_datas_ptr = result_datas.as_mut_ptr();
            let mut curve_value: C::Curve = Default::default();
            let size = std::mem::size_of::<u8>() * result_datas.len();
            unsafe {
                std::ptr::copy_nonoverlapping(result_datas_ptr, &mut curve_value as *mut C::Curve as *mut u8, size);
            }

            let result = curve_value.clone();

            result
        }

        // Caching API
        // -------------------------------------------------
        type CoeffsDescriptor<'c> = GPUMsmCoeffsDesc<'c, C>;
        type BaseDescriptor<'b> = GPUMsmBaseDesc<'b, C>;

        fn get_coeffs_descriptor<'c>(&self, coeffs: &'c [C::Scalar]) -> Self::CoeffsDescriptor<'c> {
            let mut binding = GLOBAL_DEVICE_MANAGER.lock().unwrap();
            let device_manager_handle = binding.get_handle_mut();

            let t1 = start_timer!(|| format!("init_msm_with_cached_scalars"));
            let scalars_id = device_manager_handle.init_msm_with_cached_scalars::<C>(coeffs).unwrap();
            end_timer!(t1);

            Self::CoeffsDescriptor { raw: scalars_id, _marker: PhantomData }
        }

        fn get_base_descriptor<'b>(&self, base: &'b [C]) -> Self::BaseDescriptor<'b> {
            let mut binding = GLOBAL_DEVICE_MANAGER.lock().unwrap();
            let device_manager_handle = binding.get_handle_mut();

            let t1 = start_timer!(|| format!("init_msm_with_cached_bases"));
            let bases_id = device_manager_handle.init_msm_with_cached_bases::<C>(base).unwrap();
            end_timer!(t1);

            Self::BaseDescriptor { raw: bases_id, _marker: PhantomData }
        }

        fn msm_with_cached_scalars(
            &self,
            coeffs: &Self::CoeffsDescriptor<'_>,
            base: &[C],
        ) -> C::Curve {

            let mut binding = GLOBAL_DEVICE_MANAGER.lock().unwrap();
            let device_manager_handle = binding.get_handle_mut();

            let t1 = start_timer!(|| format!("execute_msm_with_cached_scalars"));
            let mut result_datas = device_manager_handle.
                execute_msm_with_cached_scalars::<C>(coeffs.raw, base).unwrap();
            end_timer!(t1);

            let result_datas_ptr = result_datas.as_mut_ptr();
            let mut curve_value: C::Curve = Default::default();
            let size = std::mem::size_of::<u8>() * result_datas.len();
            unsafe {
                std::ptr::copy_nonoverlapping(result_datas_ptr, &mut curve_value as *mut C::Curve as *mut u8, size);
            }

            let result = curve_value.clone();

            result
        }

        fn msm_with_cached_base(
            &self,
            coeffs: &[C::Scalar],
            base: &Self::BaseDescriptor<'_>,
        ) -> C::Curve {
            
            let mut binding = GLOBAL_DEVICE_MANAGER.lock().unwrap();
            let device_manager_handle = binding.get_handle_mut();

            let t1 = start_timer!(|| format!("execute_msm_with_cached_bases"));
            let mut result_datas = device_manager_handle.
                execute_msm_with_cached_bases::<C>(coeffs, base.raw).unwrap();
            end_timer!(t1);

            let result_datas_ptr = result_datas.as_mut_ptr();
            let mut curve_value: C::Curve = Default::default();
            let size = std::mem::size_of::<u8>() * result_datas.len();
            unsafe {
                std::ptr::copy_nonoverlapping(result_datas_ptr, &mut curve_value as *mut C::Curve as *mut u8, size);
            }

            let result = curve_value.clone();

            result
        }

        fn msm_with_cached_inputs(
            &self,
            coeffs: &Self::CoeffsDescriptor<'_>,
            base: &Self::BaseDescriptor<'_>,
        ) -> C::Curve {
            let mut binding = GLOBAL_DEVICE_MANAGER.lock().unwrap();
            let device_manager_handle = binding.get_handle_mut();

            let t1 = start_timer!(|| format!("execute_msm_with_cached_input"));
            let mut result_datas = device_manager_handle.
                execute_msm_with_cached_input::<C>(coeffs.raw, base.raw).unwrap();
            end_timer!(t1);

            let result_datas_ptr = result_datas.as_mut_ptr();
            let mut curve_value: C::Curve = Default::default();
            let size = std::mem::size_of::<u8>() * result_datas.len();
            unsafe {
                std::ptr::copy_nonoverlapping(result_datas_ptr, &mut curve_value as *mut C::Curve as *mut u8, size);
            }

            let result = curve_value.clone();
            
            result
        }
    }

    // Backend-agnostic engine objects
    // ---------------------------------------------------
    #[derive(Debug)]
    pub struct PlonkEngine<C: CurveAffine, MsmEngine: MsmAccel<C>> {
        pub msm_backend: MsmEngine,
        _marker: PhantomData<C>, // compiler complains about unused C otherwise
    }

    #[derive(Default)]
    pub struct PlonkEngineConfig<C, M> {
        curve: PhantomData<C>,
        msm_backend: M,
    }

    #[derive(Default)]
    pub struct NoCurve;

    #[derive(Default)]
    pub struct HasCurve<C: CurveAffine>(PhantomData<C>);

    #[derive(Default)]
    pub struct NoMsmEngine;

    pub struct HasMsmEngine<C: CurveAffine, M: MsmAccel<C>>(M, PhantomData<C>);

    impl PlonkEngineConfig<NoCurve, NoMsmEngine> {
        pub fn new() -> PlonkEngineConfig<NoCurve, NoMsmEngine> {
            Default::default()
        }

        pub fn set_curve<C: CurveAffine>(self) -> PlonkEngineConfig<HasCurve<C>, NoMsmEngine> {
            Default::default()
        }

        pub fn build_default<C: CurveAffine + halo2curves::pairing::Engine>() -> PlonkEngine<C, GPUEngine> {
            PlonkEngine {
                msm_backend: GPUEngine::new(),
                _marker: Default::default(),
            }
        }
    }

    impl<C: CurveAffine, M> PlonkEngineConfig<HasCurve<C>, M> {
        pub fn set_msm<MsmEngine: MsmAccel<C>>(
            self,
            engine: MsmEngine,
        ) -> PlonkEngineConfig<HasCurve<C>, HasMsmEngine<C, MsmEngine>> {
            // Copy all other parameters
            let Self { curve, .. } = self;
            // Return with modified MSM engine
            PlonkEngineConfig {
                curve,
                msm_backend: HasMsmEngine(engine, Default::default()),
            }
        }
    }

    impl<C: CurveAffine, M: MsmAccel<C>> PlonkEngineConfig<HasCurve<C>, HasMsmEngine<C, M>> {
        pub fn build(self) -> PlonkEngine<C, M> {
            PlonkEngine {
                msm_backend: self.msm_backend.0,
                _marker: Default::default(),
            }
        }
    }
}

// Testing
// ---------------------------------------------------

#[cfg(test)]
mod test {
    use crate::zal::impls::{H2cEngine, PlonkEngineConfig};
    use crate::zal::traits::MsmAccel;
    use halo2curves::bn256::G1Affine;
    use halo2curves::msm::best_multiexp;
    use halo2curves::CurveAffine;

    use ark_std::{end_timer, start_timer};
    use ff::Field;
    use group::{Curve, Group};
    use rand_core::OsRng;

    use crate::zal::impls_gpu::GPUEngine;

    fn run_msm_zal_default<C: CurveAffine>(min_k: usize, max_k: usize) {
        let points = (0..1 << max_k)
            .map(|_| C::Curve::random(OsRng))
            .collect::<Vec<_>>();
        let mut affine_points = vec![C::identity(); 1 << max_k];
        C::Curve::batch_normalize(&points[..], &mut affine_points[..]);
        let points = affine_points;

        let scalars = (0..1 << max_k)
            .map(|_| C::Scalar::random(OsRng))
            .collect::<Vec<_>>();

        for k in min_k..=max_k {
            let points = &points[..1 << k];
            let scalars = &scalars[..1 << k];

            let t0 = start_timer!(|| format!("freestanding msm k={}", k));
            let e0 = best_multiexp(scalars, points);
            end_timer!(t0);

            let engine = PlonkEngineConfig::build_default::<G1Affine>();
            let t1 = start_timer!(|| format!("H2cEngine msm k={}", k));
            let e1 = engine.msm_backend.msm(scalars, points);
            end_timer!(t1);

            assert_eq!(e0, e1);

            // Caching API
            // -----------
            let t2 = start_timer!(|| format!("H2cEngine msm cached base k={}", k));
            let base_descriptor = engine.msm_backend.get_base_descriptor(points);
            let e2 = engine
                .msm_backend
                .msm_with_cached_base(scalars, &base_descriptor);
            end_timer!(t2);

            assert_eq!(e0, e2)
        }
    }

    fn run_msm_zal_custom<C: CurveAffine>(min_k: usize, max_k: usize) {
        let points = (0..1 << max_k)
            .map(|_| C::Curve::random(OsRng))
            .collect::<Vec<_>>();
        let mut affine_points = vec![C::identity(); 1 << max_k];
        C::Curve::batch_normalize(&points[..], &mut affine_points[..]);
        let points = affine_points;

        let scalars = (0..1 << max_k)
            .map(|_| C::Scalar::random(OsRng))
            .collect::<Vec<_>>();

        for k in min_k..=max_k {
            let points = &points[..1 << k];
            let scalars = &scalars[..1 << k];

            let t0 = start_timer!(|| format!("freestanding msm k={}", k));
            let e0 = best_multiexp(scalars, points);
            end_timer!(t0);

            let engine = PlonkEngineConfig::new()
                .set_curve::<G1Affine>()
                .set_msm(H2cEngine::new())
                .build();
            let t1 = start_timer!(|| format!("H2cEngine msm k={}", k));
            let e1 = engine.msm_backend.msm(scalars, points);
            end_timer!(t1);

            assert_eq!(e0, e1);

            // Caching API
            // -----------
            let t2 = start_timer!(|| format!("H2cEngine msm cached base k={}", k));
            let base_descriptor = engine.msm_backend.get_base_descriptor(points);
            let e2 = engine
                .msm_backend
                .msm_with_cached_base(scalars, &base_descriptor);
            end_timer!(t2);

            assert_eq!(e0, e2)
        }
    }

    fn run_msm_zal_gpu<C: CurveAffine>(min_k: usize, max_k: usize) {
        let points = (0..1 << max_k)
            .map(|_| C::Curve::random(OsRng))
            .collect::<Vec<_>>();

        let mut affine_points = vec![C::identity(); 1 << max_k];
        C::Curve::batch_normalize(&points[..], &mut affine_points[..]);
        let points = affine_points;

        let scalars = (0..1 << max_k)
            .map(|_| C::Scalar::random(OsRng))
            .collect::<Vec<_>>();

        for k in min_k..=max_k {
            let points = &points[..1 << k];
            let scalars = &scalars[..1 << k];

            // CPU
            let t0 = start_timer!(|| format!("freestanding msm k={}", k));
            let e0 = best_multiexp(scalars, points);
            end_timer!(t0);

            // GPU
            let engine = PlonkEngineConfig::new()
                .set_curve::<G1Affine>()
                .set_msm(GPUEngine::new())
                .build();

            let t1 = start_timer!(|| format!("H2cEngine msm k={}", k));
            let e1 = engine.msm_backend.msm(scalars, points);
            end_timer!(t1);
            assert_eq!(e0.to_affine(), e1.to_affine());
            println!("\nRun k: {}, msm compare successfully\n", k);
            
            // Caching API
            // ----------- msm with cached bases
            let t2 = start_timer!(|| format!("GPUEngine msm cached base k={}", k));
            let base_descriptor = engine.msm_backend.get_base_descriptor(points);
            let e2 = engine
                .msm_backend
                .msm_with_cached_base(scalars, &base_descriptor);
            end_timer!(t2);

            assert_eq!(e0.to_affine(), e2.to_affine());
            println!("\nRun k: {}, msm with cached bases compare successfully\n", k);

            // ----------- msm with cached scalars
            let t3 = start_timer!(|| format!("GPUEngine msm cached scalars k={}", k));
            let scalar_descriptor = engine.msm_backend.get_coeffs_descriptor(scalars);
            let e3 = engine
                .msm_backend
                .msm_with_cached_scalars(&scalar_descriptor, points);
            end_timer!(t3);

            assert_eq!(e0.to_affine(), e3.to_affine());
            println!("\nRun k: {}, msm with cached scalars compare successfully\n", k);

            // ----------- msm with cached inputs
            let t4 = start_timer!(|| format!("GPUEngine msm cached inputs k={}", k));
            let scalar_descriptor = engine.msm_backend.get_coeffs_descriptor(scalars);
            let base_descriptor = engine.msm_backend.get_base_descriptor(points);
            let e3 = engine
                .msm_backend
                .msm_with_cached_inputs(&scalar_descriptor, &base_descriptor);
            end_timer!(t4);

            assert_eq!(e0.to_affine(), e3.to_affine());
            println!("\nRun k: {}, msm with cached inputs compare successfully\n", k);
        }
    }

    #[test]
    fn test_msm_zal() {
        run_msm_zal_default::<G1Affine>(3, 14);
        run_msm_zal_custom::<G1Affine>(3, 14);
        run_msm_zal_gpu::<G1Affine>(10, 20);
    }
}
