use super::*;
use super::CurveAffine;
use panda::gpu_manager::unit::*;
use panda::gpu_manager::wrapper::*;

impl DeviceManagerContext {
    /// The core session of the MSM computation execution.
    pub fn session_msm_gpu<C: CurveAffine>(
        &mut self,
        gm: &PandaGpuManager,
        scalars: &[C::Scalar],
        bases: &[C],
    ) -> Result<Vec<u8>, DeviceManagerError> {

        // Convert scalars and bases to bytes
        let t1 = start_timer!(|| format!("[device] transmute_values"));
        let scalars_bytes = transmute_values(scalars.as_ref().as_ref());
        let bases_bytes = transmute_values(bases.as_ref().as_ref());
        end_timer!(t1);

        // Call GPU API
        let t1 = start_timer!(|| format!("[device] calling gpu api"));
        let msm_result = panda_msm_bn254_gpu(gm, scalars_bytes, bases_bytes).unwrap();
        end_timer!(t1);

        Ok(msm_result)
    }

    /// The core session of the MSM computation execution with cached bases
    pub fn session_msm_gpu_with_cached_bases<C: CurveAffine>(
        &mut self,
        gm: &PandaGpuManager,
        scalars: &[C::Scalar],
        bases_id: usize,
    ) -> Result<Vec<u8>, DeviceManagerError> {

        // Convert scalars to bytes
        let t1 = start_timer!(|| format!("[device] transmute_values"));
        let scalars_bytes = transmute_values(scalars.as_ref().as_ref());
        end_timer!(t1);

        // Call GPU API
        let t1 = start_timer!(|| format!("[device] calling gpu api"));
        let msm_result = panda_msm_bn254_gpu_with_cached_bases(gm, scalars_bytes, bases_id).unwrap();
        end_timer!(t1);

        Ok(msm_result)
    }

    /// The core session of the MSM computation execution with cached scalars.
    pub fn session_msm_gpu_with_cached_scalars<C: CurveAffine>(
        &mut self,
        gm: &PandaGpuManager,
        scalars_id: usize,
        bases: &[C],
    ) -> Result<Vec<u8>, DeviceManagerError> {

        // Convert bases to bytes
        let t1 = start_timer!(|| format!("[device] transmute_values"));
        let bases_bytes = transmute_values(bases.as_ref().as_ref());
        end_timer!(t1);

        // Call GPU API
        let t1 = start_timer!(|| format!("[device] calling gpu api"));
        let msm_result = panda_msm_bn254_gpu_with_cached_scalars(gm, scalars_id, bases_bytes).unwrap();
        end_timer!(t1);

        Ok(msm_result)
    }

    /// The core session of the MSM computation execution with cached scalars and bases.
    pub fn session_msm_gpu_with_cached_input<C: CurveAffine>(
        &mut self,
        gm: &PandaGpuManager,
        scalars_id: usize,
        bases_id: usize,
    ) -> Result<Vec<u8>, DeviceManagerError> {

        // Call GPU API
        let t1 = start_timer!(|| format!("[device] calling gpu api"));
        let msm_result = panda_msm_bn254_gpu_with_cached_input(gm, scalars_id, bases_id).unwrap();
        end_timer!(t1);

        Ok(msm_result)
    }
}
