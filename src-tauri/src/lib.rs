use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;
use tauri::Manager;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

/// Result from Python sidecar
#[derive(Debug, Serialize, Deserialize)]
pub struct SidecarResult {
    pub success: bool,
    pub data: Option<PredictionData>,
    pub error: Option<String>,
    pub traceback: Option<String>,
    pub warning: Option<String>,
    pub min_distance: Option<f64>,
    pub cif_content: Option<String>,
}

/// Prediction data from the model
#[derive(Debug, Serialize, Deserialize)]
pub struct PredictionData {
    pub num_atoms: usize,
    pub elements: Vec<i32>,
    pub element_symbols: Vec<String>,
    pub energy_grid: Vec<f64>,
    pub dos_s: Vec<Vec<f64>>,
    pub dos_p: Vec<Vec<f64>>,
    pub dos_d: Vec<Vec<f64>>,
    pub dos_f: Vec<Vec<f64>>,
    pub total_atomic_dos: Vec<Vec<f64>>,
    pub total_crystal_dos: Vec<f64>,
    pub cif_content: String,
}

/// Find sidecar executable
fn find_sidecar_exe(app: &tauri::AppHandle) -> Option<PathBuf> {
    let exe_name = if cfg!(windows) { 
        "dos-gcnn-sidecar.exe" 
    } else { 
        "dos-gcnn-sidecar" 
    };
    
    // 1. Try resource directory (for bundled app)
    if let Ok(resource_dir) = app.path().resource_dir() {
        let bundled_path = resource_dir.join(exe_name);
        if bundled_path.exists() {
            return Some(bundled_path);
        }
        // Also check in sidecar subdirectory
        let sidecar_path = resource_dir.join("sidecar").join(exe_name);
        if sidecar_path.exists() {
            return Some(sidecar_path);
        }
    }
    
    // 2. Try relative to exe (for portable/installed app)
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            // Same directory as main exe
            let local_path = exe_dir.join(exe_name);
            if local_path.exists() {
                return Some(local_path);
            }
            // In sidecar subdirectory
            let sidecar_path = exe_dir.join("sidecar").join(exe_name);
            if sidecar_path.exists() {
                return Some(sidecar_path);
            }
        }
    }
    
    // 3. Try development paths
    if let Ok(cwd) = std::env::current_dir() {
        // From project root - PyInstaller output
        let dev_path = cwd.join("python-model").join("dist").join(exe_name);
        if dev_path.exists() {
            return Some(dev_path);
        }
        // From src-tauri
        if let Some(parent) = cwd.parent() {
            let parent_path = parent.join("python-model").join("dist").join(exe_name);
            if parent_path.exists() {
                return Some(parent_path);
            }
        }
    }
    
    None
}

/// Run DOS prediction on a CIF file using sidecar executable
#[tauri::command]
async fn predict_dos(app: tauri::AppHandle, cif_path: String, force: Option<bool>) -> Result<SidecarResult, String> {
    // Validate input path exists
    let path = PathBuf::from(&cif_path);
    if !path.exists() {
        return Ok(SidecarResult {
            success: false,
            data: None,
            error: Some(format!("File not found: {}", cif_path)),
            traceback: None,
            warning: None,
            min_distance: None,
            cif_content: None,
        });
    }

    // Find sidecar executable
    let sidecar_exe = match find_sidecar_exe(&app) {
        Some(exe) => exe,
        None => {
            let exe_path = std::env::current_exe().ok();
            let cwd = std::env::current_dir().ok();
            let resource_dir = app.path().resource_dir().ok();
            
            return Ok(SidecarResult {
                success: false,
                data: None,
                error: Some(format!(
                    "Sidecar executable not found.\nSearched:\n- resource_dir: {:?}\n- exe_dir: {:?}\n- cwd: {:?}",
                    resource_dir, exe_path, cwd
                )),
                traceback: None,
                warning: None,
                min_distance: None,
                cif_content: None,
            });
        }
    };

    // Run sidecar executable (hidden window on Windows)
    #[cfg(windows)]
    const CREATE_NO_WINDOW: u32 = 0x08000000;
    
    let mut cmd = Command::new(&sidecar_exe);
    cmd.arg(&cif_path);
    
    // Add --force flag if requested
    if force.unwrap_or(false) {
        cmd.arg("--force");
    }
    
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    
    let output = cmd.output()
        .map_err(|e| format!("Failed to run sidecar: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        
        // Try to parse error from stdout (sidecar outputs JSON even on error)
        if let Some(json_start) = stdout.find('{') {
            if let Ok(result) = serde_json::from_str::<SidecarResult>(&stdout[json_start..]) {
                return Ok(result);
            }
        }
        
        return Ok(SidecarResult {
            success: false,
            data: None,
            error: Some(format!(
                "Sidecar failed (exit code: {:?}).\nStderr: {}\nStdout: {}",
                output.status.code(),
                stderr,
                stdout
            )),
            traceback: None,
            warning: None,
            min_distance: None,
            cif_content: None,
        });
    }

    // Parse JSON output
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Find JSON in output (skip any startup messages)
    let json_start = stdout.find('{');
    let json_str = match json_start {
        Some(pos) => &stdout[pos..],
        None => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Ok(SidecarResult {
                success: false,
                data: None,
                error: Some(format!(
                    "No JSON found in sidecar output.\nStdout: {}\nStderr: {}",
                    stdout, stderr
                )),
                traceback: None,
                warning: None,
                min_distance: None,
                cif_content: None,
            });
        }
    };
    
    let result: SidecarResult = serde_json::from_str(json_str)
        .map_err(|e| format!("Failed to parse JSON: {}.\nRaw output: {}", e, json_str))?;

    Ok(result)
}

/// Check if sidecar is available
#[tauri::command]
async fn check_sidecar(app: tauri::AppHandle) -> Result<bool, String> {
    Ok(find_sidecar_exe(&app).is_some())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![predict_dos, check_sidecar])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
