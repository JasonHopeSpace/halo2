use super::CurveAffine;
use group::{GroupOpsOwned, ScalarMulOwned};
use halo2curves::ff::Field;

use super::*;
use panda::gpu_manager::*;

use lazy_static::lazy_static;
use std::sync::Mutex;

///
pub trait FftGroup<Scalar: Field>:
    Copy + Send + Sync + 'static + GroupOpsOwned + ScalarMulOwned<Scalar>
{
}

///
impl<T, Scalar> FftGroup<Scalar> for T
where
    Scalar: Field,
    T: Copy + Send + Sync + 'static + GroupOpsOwned + ScalarMulOwned<Scalar>,
{
}

lazy_static! {
    ///
    pub static ref GLOBAL_DEVICE_MANAGER: Mutex<DeviceManager> = Mutex::new(DeviceManager::new());
}

///
#[derive(Clone, Debug)]
pub struct DeviceManager {
    ///
    pub handle: Box<DeviceManagerContext>,
}

/// The implement of DeviceManager
impl DeviceManager {
    /// Create
    pub fn new() -> Self {
        let context = DeviceManagerContext {
            gpu_device_num: 0,
            actived_device_num: 0,
            devices: Vec::<DeviceUnit>::new(),
            msm_param_uints: Vec::<MSMParamUnit>::new(),
            msm_param_uints_new: Vec::<MSMParamUnitNew>::new(),
            ntt_param_uints: Vec::<NTTParamUnit>::new(),
            init_flag: false,
            gms: Vec::<PandaGpuManager>::new(),
            gm_initialized: false,
        };
        Self {
            handle: Box::new(context),
        }
    }

    /// Get the handle of DeviceManager
    pub fn get_handle(&self) -> &DeviceManagerContext {
        &*self.handle
    }

    /// Get the mutable reference handle of DeviceManager.
    pub fn get_handle_mut(&mut self) -> &mut DeviceManagerContext {
        &mut self.handle
    }
}

///
#[derive(Clone, Debug)]
pub struct DeviceManagerContext {
    ///
    pub gpu_device_num: usize,
    ///
    pub actived_device_num: usize,
    ///
    pub devices: Vec<DeviceUnit>,
    ///
    pub gms: Vec<PandaGpuManager>,
    ///
    pub gm_initialized: bool,
    ///
    pub msm_param_uints: Vec<MSMParamUnit>,
    ///
    pub msm_param_uints_new: Vec<MSMParamUnitNew>,
    ///
    pub ntt_param_uints: Vec<NTTParamUnit>,
    ///
    pub init_flag: bool,
}

impl DeviceManagerContext {
    /// Get device and create gm
    pub fn new(
        &mut self,
    ) -> Result<(), DeviceManagerError> {

        // Get the number of GPUs
        self.gpu_device_num = self.get_gpu_device_number().unwrap();

        // In case the number of GPUs is 0, return.
        if self.gpu_device_num == 0 {
            return Err(DeviceManagerError::DeviceManagerErrorGetDeviceNum);
        }

        self.actived_device_num = self.gpu_device_num;

        if self.gm_initialized {
            return Ok(());
        }
        else
        {
            // Create the handle of gpu manager
            let gm = PandaGpuManager::new(0).unwrap();
            self.gms.push(gm);
            self.gm_initialized = true;
        }

        Ok(())
    }

    /// Pre-load bases or scalars data to device for msm
    pub fn init_msm_with_cached_bases<C: CurveAffine>(
        &mut self,
        bases: &[C],
        // scalars: Option<&[C::Scalar]>,
    ) -> Result<usize, DeviceManagerError> {

        if self.gm_initialized == false {
            self.new().unwrap();
        }
        else {
            self.new().unwrap();
        }

        // cached bases
        let mut msm_param_uint_new = MSMParamUnitNew {
            is_cached_bases: false,
            is_cached_scalars: false,
            bases_id: 0,
            scalars_id: 0,
            gm: self.gms[0].clone(),
        };

        let bases_bytes = transmute_values(bases.as_ref().as_ref());
        let d_bases_ptrs = PandaGpuManager::init_msm_cached_bases(bases_bytes).unwrap();

        self.gms[0].d_bases.push(d_bases_ptrs);
        let bases_id = self.gms[0].d_bases.len() - 1;
        msm_param_uint_new.bases_id = bases_id;
        msm_param_uint_new.is_cached_bases = true;

        // cached scalars   todo!
        self.msm_param_uints_new.push(msm_param_uint_new);

        let device_id = 0;
        // Generate new device unit of MSM.
        let device: DeviceUnit = DeviceUnit {
            device_id,
            device_type: DeviceType::DeviceTypeGPU,
            device_unit_type: DeviceUnitType::DeviceUnitTypeMSM,
            device_status: DeviceStatusType::DeviceStatusReady,
        };
        self.devices.push(device);

        Ok(bases_id)
    }

    /// Pre-load bases or scalars data to device for msm
    pub fn init_msm_with_cached_scalars<C: CurveAffine>(
        &mut self,
        scalars: &[C::Scalar],
    ) -> Result<usize, DeviceManagerError> {

        if self.gm_initialized == false {
            self.new().unwrap();
        }
        else {
            self.new().unwrap();
        }

        // cached bases
        let mut msm_param_uint_new = MSMParamUnitNew {
            is_cached_bases: false,
            is_cached_scalars: false,
            bases_id: 0,
            scalars_id: 0,
            gm: self.gms[0].clone(),
        };

        let scalars_bytes = transmute_values(scalars.as_ref().as_ref());
        let d_scalars_ptrs = PandaGpuManager::init_msm_cached_scalars(scalars_bytes).unwrap();

        self.gms[0].d_scalars.push(d_scalars_ptrs);

        let scalars_id = self.gms[0].d_scalars.len() - 1;
        msm_param_uint_new.scalars_id = scalars_id;
        msm_param_uint_new.is_cached_bases = true;

        self.gms[0].scalars_len.push(scalars_bytes.len());
        self.msm_param_uints_new.push(msm_param_uint_new);

        let device_id = 0;
        // Generate new device unit of MSM.
        let device: DeviceUnit = DeviceUnit {
            device_id,
            device_type: DeviceType::DeviceTypeGPU,
            device_unit_type: DeviceUnitType::DeviceUnitTypeMSM,
            device_status: DeviceStatusType::DeviceStatusReady,
        };
        self.devices.push(device);

        Ok(scalars_id)
    }

    /// Initialize
    pub fn init_all(
        &mut self,
        init_device_unit_type: DeviceInitUnitType,
        param_id: Option<usize>,
        bases: Option<&[&[u8]]>,
        omega: Option<&[u8]>,
    ) -> Result<(), DeviceManagerError> {
        // Get the number of GPUs
        self.gpu_device_num = self.get_gpu_device_number().unwrap();

        // In case the number of GPUs is 0, return.
        if self.gpu_device_num == 0 {
            return Err(DeviceManagerError::DeviceManagerErrorGetDeviceNum);
        }

        // Mapping initialization of device computation types.
        let init_uint_type = match init_device_unit_type {
            DeviceInitUnitType::DeviceInitUnitTypeNone => {
                PandaGpuManagerInitUnitType::PandaGpuManagerInitUnitTypeNone
            }
            DeviceInitUnitType::DeviceInitUnitTypeMSM => {
                PandaGpuManagerInitUnitType::PandaGpuManagerInitUnitTypeMSM
            }
            DeviceInitUnitType::DeviceInitUnitTypeNTT => {
                PandaGpuManagerInitUnitType::PandaGpuManagerInitUnitTypeNTT
            }
            DeviceInitUnitType::DevicerInitUnitTypeALL => {
                PandaGpuManagerInitUnitType::PandaGpuManagerInitUnitTypeALL
            }
            DeviceInitUnitType::DeviceInitUnitTypeEvaluation => todo!(),
        };

        // init
        for device_id in 0..self.gpu_device_num {
            // GPU init and get the handle of gpu manager. Setup and copy bases data
            let gm = PandaGpuManager::init_all(0, init_uint_type.clone(), bases, omega).unwrap();

            match init_uint_type {
                PandaGpuManagerInitUnitType::PandaGpuManagerInitUnitTypeNone => todo!(),
                PandaGpuManagerInitUnitType::PandaGpuManagerInitUnitTypeMSM => {
                    if let Some(id) = param_id {
                        let msm_param_uint = MSMParamUnit {
                            param_id: id,
                            in_usze: true,
                            init_flag: true,
                            gm,
                        };
                        self.msm_param_uints.push(msm_param_uint);
                        // Generate new device unit of MSM.
                        let device: DeviceUnit = DeviceUnit {
                            device_id,
                            device_type: DeviceType::DeviceTypeGPU,
                            device_unit_type: DeviceUnitType::DeviceUnitTypeMSM,
                            device_status: DeviceStatusType::DeviceStatusReady,
                        };
                        self.devices.push(device);
                    } else {
                        return Err(DeviceManagerError::DeviceManagerErrorGetDeviceNum);
                    }
                }
                PandaGpuManagerInitUnitType::PandaGpuManagerInitUnitTypeNTT => {
                    let ntt_param_uint = NTTParamUnit {
                        in_usze: true,
                        init_flag: true,
                        gm,
                    };

                    self.ntt_param_uints.push(ntt_param_uint);
                    // Generate new device unit of NTT.
                    let device: DeviceUnit = DeviceUnit {
                        device_id,
                        device_type: DeviceType::DeviceTypeGPU,
                        device_unit_type: DeviceUnitType::DeviceUnitTypeNTT,
                        device_status: DeviceStatusType::DeviceStatusReady,
                    };
                    self.devices.push(device);
                }
                PandaGpuManagerInitUnitType::PandaGpuManagerInitUnitTypeALL => {
                    if let Some(id) = param_id {
                        let msm_param_uint = MSMParamUnit {
                            param_id: id,
                            in_usze: true,
                            init_flag: true,
                            gm: gm.clone(),
                        };
                        self.msm_param_uints.push(msm_param_uint);
                        // Generate new device unit of MSM.
                        let device: DeviceUnit = DeviceUnit {
                            device_id,
                            device_type: DeviceType::DeviceTypeGPU,
                            device_unit_type: DeviceUnitType::DeviceUnitTypeMSM,
                            device_status: DeviceStatusType::DeviceStatusReady,
                        };
                        self.devices.push(device);
                    } else {
                        return Err(DeviceManagerError::DeviceManagerErrorGetDeviceNum);
                    }
                    // Generate new device unit of NTT.
                    let ntt_param_uint = NTTParamUnit {
                        in_usze: true,
                        init_flag: true,
                        gm: gm.clone(),
                    };

                    self.ntt_param_uints.push(ntt_param_uint);

                    let device: DeviceUnit = DeviceUnit {
                        device_id,
                        device_type: DeviceType::DeviceTypeGPU,
                        device_unit_type: DeviceUnitType::DeviceUnitTypeNTT,
                        device_status: DeviceStatusType::DeviceStatusReady,
                    };
                    self.devices.push(device);
                }
            }
        }

        // Set actived device number and may be a need to use when performing calculations.
        self.actived_device_num = self.gpu_device_num;
        self.init_flag = true;

        Ok(())
    }

    /// Deinitialization
    pub fn deinit(&mut self) -> Result<(), DeviceManagerError> {
        // Set the GPU and active device numbers to 0 to indicate deinitialization.
        self.gpu_device_num = 0;
        self.actived_device_num = 0;

        // Clear the device lists and flags.
        for msm_param_uint in self.msm_param_uints.iter() {
            let mut gm = msm_param_uint.gm.clone();
            gm.deinit();
        }
        self.msm_param_uints.clear();
        for ntt_param_uint in self.ntt_param_uints.iter() {
            let mut gm = ntt_param_uint.gm.clone();
            gm.deinit();
        }
        self.ntt_param_uints.clear();
        self.devices.clear();
        self.init_flag = false;

        Ok(())
    }

    ///
    pub fn get_device_id(&mut self) -> Result<usize, DeviceManagerError> {
        self.gpu_device_num = self.get_gpu_device_number().unwrap();
        if self.gpu_device_num == 0 {
            return Err(DeviceManagerError::DeviceManagerErrorGetDeviceNum);
        }
        Ok(self.gpu_device_num)
    }

    /// Get available devices.
    fn get_available_device(&mut self) -> Result<usize, DeviceManagerError> {
        for i in 0..self.actived_device_num {
            match self.devices[i].device_status {
                DeviceStatusType::DeviceStatusNone => todo!(),
                DeviceStatusType::DeviceStatusIdle => todo!(),
                DeviceStatusType::DeviceStatusReady => {
                    return Ok(i);
                }
                DeviceStatusType::DeviceStatusRunning => todo!(),
            }
        }
        Ok(NO_AVAILABE_DEVICE)
    }

    /// Run the MSM calculation process.
    pub fn execute_msm<C: CurveAffine>(
        &mut self,
        scalars: &[C::Scalar],
        bases: &[C],
    ) -> Result<Vec<u8>, DeviceManagerError> {

        let t1 = start_timer!(|| format!("execute_msm init"));
        if self.gm_initialized == false {
            self.new().unwrap();
        }
        else {
            self.new().unwrap();
        }
        end_timer!(t1);

        let gm = &mut self.gms[0].clone();
        gm.set_config(PandaMSMResultCoordinateType::Projective);

        let t1 = start_timer!(|| format!("session_msm_gpu"));
        let msm_result = self.session_msm_gpu::<C>(gm, scalars, bases).unwrap();
        end_timer!(t1);

        Ok(msm_result)
    }

    pub fn execute_msm_with_cached_bases<C: CurveAffine>(
        &mut self,
        scalars: &[C::Scalar],
        bases_id: usize,
    ) -> Result<Vec<u8>, DeviceManagerError> {

        let t1 = start_timer!(|| format!("execute_msm_with_cached_bases init"));
        if self.gm_initialized == false {
            self.new().unwrap();
        }
        else {
            self.new().unwrap();
        }
        end_timer!(t1);
        
        let gm = &mut self.gms[0].clone();
        gm.set_config(PandaMSMResultCoordinateType::Projective);

        let t1 = start_timer!(|| format!("session_msm_gpu_with_cached_bases"));
        let msm_result = self.session_msm_gpu_with_cached_bases::<C>(gm, scalars, bases_id).unwrap();
        end_timer!(t1);

        Ok(msm_result)
    }

    pub fn execute_msm_with_cached_scalars<C: CurveAffine>(
        &mut self,
        scalars_id: usize,
        bases: &[C],
    ) -> Result<Vec<u8>, DeviceManagerError> {

        let t1 = start_timer!(|| format!("execute_msm_with_cached_scalars init"));
        if self.gm_initialized == false {
            self.new().unwrap();
        }
        else {
            self.new().unwrap();
        }
        end_timer!(t1);
        
        let gm = &mut self.gms[0].clone();
        gm.set_config(PandaMSMResultCoordinateType::Projective);

        let t1 = start_timer!(|| format!("session_msm_gpu_with_cached_scalars"));
        let msm_result = self.session_msm_gpu_with_cached_scalars::<C>(gm, scalars_id, bases).unwrap();
        end_timer!(t1);

        Ok(msm_result)
    }

    pub fn execute_msm_with_cached_input<C: CurveAffine>(
        &mut self,
        scalars_id: usize,
        bases_id: usize,
    ) -> Result<Vec<u8>, DeviceManagerError> {

        let t1 = start_timer!(|| format!("execute_msm_with_cached_input init"));

        if self.gm_initialized == false {
            self.new().unwrap();
        }
        else {
            self.new().unwrap();
        }
        end_timer!(t1);
        
        let gm = &mut self.gms[0].clone();
        gm.set_config(PandaMSMResultCoordinateType::Projective);

        let t1 = start_timer!(|| format!("session_msm_gpu_with_cached_input"));
        let msm_result = self.session_msm_gpu_with_cached_input::<C>(gm, scalars_id, bases_id).unwrap();
        end_timer!(t1);

        Ok(msm_result)
    }

    /// Get the numbere of units of GPU.
    pub fn get_gpu_unit_number(&mut self) -> Result<usize, DeviceManagerError> {
        return Ok(self.devices.len());
    }

    /// Get the numbere of MSM param units of GPU.
    pub fn get_gpu_msm_param_uints_number(&mut self) -> Result<usize, DeviceManagerError> {
        return Ok(self.msm_param_uints.len());
    }

    /// Get device info of GPUs.
    pub fn get_gpu_device_info(
        &mut self,
        device_id: usize,
    ) -> Result<DeviceGPUInfo, DeviceManagerError> {
        if self.gpu_device_num == 0 || device_id >= self.gpu_device_num {
            return Err(DeviceManagerError::DeviceManagerErrorGetDeviceNum);
        }

        let panda_gpu_info = panda::gpu_manager::device_info(device_id.try_into().unwrap());

        match panda_gpu_info {
            Ok(panda_gpu_info) => {
                let device_info = DeviceGPUInfo {
                    gpu_device_id: device_id,
                    panda_gpu_info,
                };
                Ok(device_info)
            }
            Err(_) => Err(DeviceManagerError::DeviceManagerErrorGetDeviceInfo),
        }
    }

    /// Get device infos of GPUs.
    pub fn get_gpu_device_infos(&mut self) -> Result<Vec<DeviceGPUInfo>, DeviceManagerError> {
        if self.gpu_device_num == 0 {
            return Err(DeviceManagerError::DeviceManagerErrorGetDeviceNum);
        }

        let device_infos: Vec<DeviceGPUInfo> = (0..self.gpu_device_num)
            .map(
                |id| match panda::gpu_manager::device_info(id.try_into().unwrap()) {
                    Ok(panda_gpu_info) => Ok(DeviceGPUInfo {
                        gpu_device_id: id,
                        panda_gpu_info,
                    }),
                    Err(_) => Err(DeviceManagerError::DeviceManagerErrorGetDeviceInfo),
                },
            )
            .collect::<Result<_, DeviceManagerError>>()?;

        Ok(device_infos)
    }

    ///Get device number of GPUs.
    pub fn get_gpu_device_number(&mut self) -> Result<usize, DeviceManagerError> {
        Ok(panda::gpu_manager::get_device_number()
            .unwrap()
            .try_into()
            .unwrap())
    }

    /// Set device for GPU.
    pub fn set_gpu_device(&mut self, device_id: usize) -> Result<(), DeviceManagerError> {
        if let Err(_) = panda::gpu_manager::set_device(device_id) {
            return Err(DeviceManagerError::DeviceManagerSetDeviceError);
        }
        Ok(())
    }

    /// Transmute ther formats into a byte stream.
    pub fn transmute_values<'a, U: std::fmt::Debug>(&mut self, values: &'a [U]) -> &'a [u8] {
        transmute_values(values)
    }

}
