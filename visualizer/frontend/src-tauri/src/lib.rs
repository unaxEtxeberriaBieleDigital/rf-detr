use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;

use tauri::{AppHandle, Manager, RunEvent, Runtime};

struct BackendProcess(Mutex<Option<Child>>);

fn backend_executable<R: Runtime>(
    app: &AppHandle<R>,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let executable = app
        .path()
        .resource_dir()?
        .join("backend")
        .join("visualizer-backend.exe");

    if !executable.is_file() {
        return Err(format!(
            "The packaged backend was not found at '{}'. Rebuild with visualizer\\build_visualizer.ps1.",
            executable.display()
        )
        .into());
    }

    Ok(executable)
}

fn start_backend<R: Runtime>(app: &AppHandle<R>) -> Result<(), Box<dyn std::error::Error>> {
    let mut command = Command::new(backend_executable(app)?);

    // PyInstaller builds the backend as a console application, so without this flag Windows
    // opens a stray terminal window next to the packaged app.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }

    let child = command.spawn()?;
    app.manage(BackendProcess(Mutex::new(Some(child))));
    Ok(())
}

fn stop_backend<R: Runtime>(app: &AppHandle<R>) {
    let Some(backend) = app.try_state::<BackendProcess>() else {
        return;
    };
    let Ok(mut child) = backend.0.lock() else {
        eprintln!("Could not acquire the packaged backend process lock.");
        return;
    };
    let Some(mut child) = child.take() else {
        return;
    };

    if let Err(error) = child.kill() {
        eprintln!("Could not stop the packaged backend: {error}");
    }
    if let Err(error) = child.wait() {
        eprintln!("Could not wait for the packaged backend to exit: {error}");
    }
}

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet])
        .setup(|app| {
            if !cfg!(debug_assertions) {
                start_backend(app.handle())?;
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                stop_backend(app);
            }
        });
}
