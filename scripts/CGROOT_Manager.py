import os
import sys
import subprocess
import datetime
from pathlib import Path
import platform
import shutil

def check_and_install_dependencies():
    required_modules = ['colorama']
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    if missing_modules:
        print("Missing dependencies detected: ", ", ".join(missing_modules))
        print("Attempting to install missing modules from requirements.txt...")
        try:
            req_file = Path(__file__).resolve().parent.parent / "requirements.txt"
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
        except subprocess.CalledProcessError:
            print("Automatic installation failed. Please run 'pip install -r requirements.txt' manually.")
            sys.exit(1)
        print("Installation complete. Restarting the script...\n")
        os.execv(sys.executable, [sys.executable] + sys.argv)

check_and_install_dependencies()

from colorama import init, Fore, Style
init(autoreset=True)

RED = Fore.RED
GREEN = Fore.GREEN
YELLOW = Fore.YELLOW
BLUE = Fore.BLUE
MAGENTA = Fore.MAGENTA
CYAN = Fore.CYAN
WHITE = Fore.WHITE
RESET = Style.RESET_ALL

# Determine script and project root (assuming script runs from 'scripts' subfolder)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
os.chdir(project_root)  # Change working dir to project root for all relative paths

project_name = "CGROOT++"
build_dir = project_root / "build"
log_file = project_root / "build_log.txt"

# Windows-specific compiler CMake paths
username = os.getenv("USERNAME") or os.getenv("USER") or ""
cmake_paths = {
    "Original Version": Path(r"C:\Program Files\CMake\bin\cmake.exe"),
    "Visual Studio 16 2019": Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"),
    "Visual Studio 17 2022 (Build Tools)": Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"),
    "Visual Studio 17 2022 (Community)": Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"),
    "MinGW (CMake)": Path(r"C:\mingw64\bin\cmake.exe"),
    "MSYS2 (CMake)": Path(r"C:\msys64\mingw64\bin\cmake.exe"),
    "Android SDK CMake": Path(f"C:\\Users\\{username}\\AppData\\Local\\Android\\Sdk\\cmake\\3.31.6\\bin\\cmake.exe"),
    "Standalone CMake": Path(r"C:\Program Files\CMake\bin\cmake.exe"),
}

make_tools = {
    "avr-make": Path(r"C:\avr-gcc\bin\make.exe"),
    "msys-make": Path(r"C:\msys64\usr\bin\make.exe"),
    "mingw-make": Path(r"C:\mingw64\bin\mingw32-make.exe"),
    "vs-nmake": Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\nmake.exe"),
}

def log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} - {message}\n")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_os_type():
    os_type = platform.system()
    if os_type == "Windows":
        return "windows"
    else:
        return "linux"

def get_cmake_and_make_paths():
    os_type = get_os_type()
    if os_type == "windows":
        # On Windows, prompt user to select compiler from cmake_paths later
        return "windows"
    else:
        # On Linux/macOS: just verify cmake and make in PATH
        if shutil.which("cmake") is None:
            print(f"{RED}ERROR: 'cmake' not found in PATH. Please install CMake.{RESET}")
            sys.exit(1)
        if shutil.which("make") is None:
            print(f"{YELLOW}WARNING: 'make' not found in PATH. You may need to install build-essential or equivalent.{RESET}")
        return "linux"

def find_windeployqt():
    """
    Find windeployqt.exe from Qt6 installation.
    Searches common Qt6 installation paths on Windows.
    Returns Path to windeployqt.exe or None if not found.
    """
    if get_os_type() != "windows":
        return None
    
    # Common Qt6 installation paths on Windows
    qt6_paths = [
        # Qt6 installed via installer (default location)
        Path(r"C:\Qt\6.10.0\msvc2019_64\bin\windeployqt.exe"),
        Path(r"C:\Qt\6.9.0\msvc2019_64\bin\windeployqt.exe"),
        Path(r"C:\Qt\6.8.0\msvc2019_64\bin\windeployqt.exe"),
        Path(r"C:\Qt\6.7.0\msvc2019_64\bin\windeployqt.exe"),
        Path(r"C:\Qt\6.6.0\msvc2019_64\bin\windeployqt.exe"),
        # Visual Studio 2022 (msvc2022_64)
        Path(r"C:\Qt\6.10.0\msvc2022_64\bin\windeployqt.exe"),
        Path(r"C:\Qt\6.9.0\msvc2022_64\bin\windeployqt.exe"),
        Path(r"C:\Qt\6.8.0\msvc2022_64\bin\windeployqt.exe"),
        Path(r"C:\Qt\6.7.0\msvc2022_64\bin\windeployqt.exe"),
        Path(r"C:\Qt\6.6.0\msvc2022_64\bin\windeployqt.exe"),
        # MinGW
        Path(r"C:\Qt\6.10.0\mingw_64\bin\windeployqt.exe"),
        Path(r"C:\Qt\6.9.0\mingw_64\bin\windeployqt.exe"),
        Path(r"C:\Qt\6.8.0\mingw_64\bin\windeployqt.exe"),
        # User-specific installation
        Path.home() / "Qt" / "6.10.0" / "msvc2019_64" / "bin" / "windeployqt.exe",
        Path.home() / "Qt" / "6.9.0" / "msvc2019_64" / "bin" / "windeployqt.exe",
        Path.home() / "Qt" / "6.10.0" / "msvc2022_64" / "bin" / "windeployqt.exe",
        Path.home() / "Qt" / "6.9.0" / "msvc2022_64" / "bin" / "windeployqt.exe",
    ]
    
    # Check if windeployqt is in PATH
    windeployqt_in_path = shutil.which("windeployqt.exe")
    if windeployqt_in_path:
        return Path(windeployqt_in_path)
    
    # Search common installation paths
    for qt_path in qt6_paths:
        if qt_path.exists():
            return qt_path
    
    # Try to find Qt6 installation directory by searching for Qt6Config.cmake
    # This is a more robust method that works with CMAKE_PREFIX_PATH
    try:
        # Check if CMAKE_PREFIX_PATH environment variable is set
        cmake_prefix = os.getenv("CMAKE_PREFIX_PATH", "")
        if cmake_prefix:
            prefix_path = Path(cmake_prefix.split(os.pathsep)[0])
            windeployqt = prefix_path / "bin" / "windeployqt.exe"
            if windeployqt.exists():
                return windeployqt
        
        # Search in Program Files
        program_files = Path(os.getenv("ProgramFiles", r"C:\Program Files"))
        for qt_dir in program_files.glob("Qt/*/msvc*_64/bin/windeployqt.exe"):
            if qt_dir.exists():
                return qt_dir
    except Exception:
        pass
    
    return None


def find_ninja_tool():
    """
    Find ninja.exe in common locations.
    """
    # Check PATH first
    if shutil.which("ninja"):
        return "ninja"

    # Common locations
    search_paths = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"),
    ]
    
    for path in search_paths:
        if path.exists():
            return str(path)
            
    return None

def get_current_generator():
    """Reads CMakeCache.txt to find current generator"""
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.exists():
        return None
        
    try:
        with open(cache_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("CMAKE_GENERATOR:INTERNAL="):
                    return line.split("=")[1].strip()
    except:
        pass
    return None

def find_vcvars_bat():
    """Find vcvars64.bat to set up MSVC environment for Ninja"""
    search_paths = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"),
    ]
    for path in search_paths:
        if path.exists():
            return str(path)
    return None

def deploy_qt_dlls(gui_executable, configuration):
    """
    Deploy Qt DLLs for the GUI executable using windeployqt.
    
    Steps:
    1. Create a separate folder (e.g., Debug/GuiApp or Release/GuiApp)
    2. Copy the GUI executable to that folder
    3. Run windeployqt on the copied executable
    4. Return the path to the deployed executable
    
    Args:
        gui_executable: Path to the original GUI executable
        configuration: Build configuration (Debug or Release)
    
    Returns:
        Path to the deployed GUI executable, or None if deployment failed
    """
    if get_os_type() != "windows":
        # On non-Windows, just return the original path
        return gui_executable
    
    if not gui_executable.exists():
        print(f"{RED}ERROR: GUI executable not found: {gui_executable}{RESET}")
        return None
    
    # Find windeployqt.exe
    windeployqt = find_windeployqt()
    if not windeployqt or not windeployqt.exists():
        print(f"{YELLOW}WARNING: windeployqt.exe not found. Qt DLLs will not be deployed.{RESET}")
        print(f"{YELLOW}Please ensure Qt6 is installed and windeployqt.exe is in PATH or set CMAKE_PREFIX_PATH.{RESET}")
        return gui_executable
    
    print(f"{BLUE}Found windeployqt: {windeployqt}{RESET}")
    
    # Step 1: Create separate folder for GUI app (e.g., build/bin/Debug/GuiApp)
    gui_app_dir = build_dir / "bin" / configuration / "GuiApp"
    gui_app_dir.mkdir(parents=True, exist_ok=True)
    print(f"{BLUE}Created GUI app directory: {gui_app_dir}{RESET}")
    
    # Step 2: Copy GUI executable to the separate folder
    deployed_executable = gui_app_dir / gui_executable.name
    try:
        shutil.copy2(gui_executable, deployed_executable)
        print(f"{GREEN}Copied GUI executable to: {deployed_executable}{RESET}")
    except Exception as e:
        print(f"{RED}ERROR: Failed to copy GUI executable: {e}{RESET}")
        return None
    
    # Step 3: Run windeployqt on the copied executable
    # Use --qmldir if you have QML files, --release for Release builds
    print(f"{BLUE}Deploying Qt DLLs using windeployqt...{RESET}")
    windeployqt_args = [
        str(windeployqt),
        "--compiler-runtime",  # Include Visual C++ runtime
        "--force",  # Overwrite existing files
    ]
    
    # Add --release flag for Release builds
    if configuration.lower() == "release":
        windeployqt_args.append("--release")
    
    # Add the executable path (use quotes to handle spaces)
    windeployqt_args.append(str(deployed_executable))
    
    try:
        # Run windeployqt with proper path handling
        result = subprocess.run(
            windeployqt_args,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        if result.returncode == 0:
            print(f"{GREEN}Successfully deployed Qt DLLs to: {gui_app_dir}{RESET}")
            if result.stdout:
                # Show windeployqt output if verbose
                print(f"{CYAN}windeployqt output:{RESET}")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        print(f"  {line}")
            return deployed_executable
        else:
            print(f"{RED}ERROR: windeployqt failed with exit code {result.returncode}{RESET}")
            if result.stderr:
                print(f"{RED}Error output:{RESET}")
                print(result.stderr)
            if result.stdout:
                print(f"{YELLOW}Standard output:{RESET}")
                print(result.stdout)
            # Return the executable anyway - it might still work
            return deployed_executable
    except subprocess.TimeoutExpired:
        print(f"{RED}ERROR: windeployqt timed out{RESET}")
        return deployed_executable
    except Exception as e:
        print(f"{RED}ERROR: Failed to run windeployqt: {e}{RESET}")
        return deployed_executable

def detect_compilers():
    os_type = get_cmake_and_make_paths()
    if os_type == "windows":
        available = {}
        idx = 1
        print()
        print(f"{CYAN}================================================{RESET}")
        print(f"{CYAN}           {project_name} Project Manager{RESET}")
        print(f"{CYAN}================================================{RESET}")
        print()
        print(f"{GREEN}Detecting available compilers...{RESET}")
        print()
        for name, path in cmake_paths.items():
            if path.exists():
                available[idx] = (name, path)
                print(f"[{idx}] {GREEN}{name}{RESET} found at {MAGENTA}{path}{RESET}")
                idx += 1
        print()
        if not available:
            print(f"{RED}ERROR: No supported compiler toolchains found!{RESET}")
            print(f"{CYAN}Please install one of:")
            print("  - Visual Studio Build Tools 2019 or 2022")
            print("  - Visual Studio Community 2022")
            print("  - MinGW or MSYS2")
            print("  - Standalone CMake" + RESET)
            input("Press Enter to exit...")
            sys.exit(1)
        return available
    else:
        # On Linux: single option, system cmake in PATH
        return {1: ("System CMake", "cmake")}

def detect_make_tool():
    os_type = get_os_type()
    if os_type == "windows":
        for name, path in make_tools.items():
            if path.exists():
                return path
        return None
    else:
        if shutil.which("make") is not None:
            return "make"
        else:
            return None

def choose_compiler(available):
    if "--build" in sys.argv and len(available) > 0:
         # Auto-select first available compiler
         first_key = list(available.keys())[0]
         return available[first_key]

    while True:
        choice = input(f"{YELLOW}Enter compiler choice (1-{len(available)}): {RESET}").strip()
        if not choice.isdigit():
            print(f"{RED}Invalid input. Enter digits only.{RESET}")
            continue
        choice_int = int(choice)
        if choice_int < 1 or choice_int > len(available):
            print(f"{RED}Invalid choice. Enter a number between 1 and {len(available)}.{RESET}")
            continue
        return available[choice_int]

def pause():
    if "--build" in sys.argv: return
    input("Press Enter to continue...")

def run_command(cmd, log_append=True):
    print(f"{BLUE}Running command:{RESET} {cmd}")
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(log_file, "a", encoding="utf-8") as logf:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            print(line, end="")
            if log_append:
                logf.write(line)
    proc.wait()
    return proc.returncode

import stat
import errno

def remove_readonly(func, path, exc):
    excvalue = exc[1]
    # Check if correct exception (PermissionError usually corresponds to EACCES on Windows)
    if func in (os.rmdir, os.remove, os.unlink) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise

def kill_zombie_processes():
    """
    Kills common build processes and project executables that might lock files.
    """
    if get_os_type() != "windows":
        return

    # List of process names to kill
    targets = [
        "cmake.exe", "ninja.exe", "msbuild.exe", "make.exe",
        "cgrunner.exe", "test_diagnostics.exe", "cgroot_gui.exe"
    ]

    print(f"{YELLOW}Checking for zombie processes...{RESET}")
    for target in targets:
        try:
            # check if process exists first to avoid spamming errors? 
            # taskkill /IM <image> /F suppresses output if not found ? No, it errors.
            # We redirect stderr to null to suppress "process not found" errors.
            subprocess.run(f"taskkill /F /IM {target}", shell=True, 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

import time

def clean_build_dir():
    # Attempt to kill potential locking processes first
    kill_zombie_processes()

    if build_dir.exists() and build_dir.is_dir():
        print(f"{YELLOW}Cleaning build directory...{RESET}")
        
        # Retry logic for persistent locks
        max_retries = 3
        for i in range(max_retries):
            try:
                shutil.rmtree(build_dir, onerror=remove_readonly)
                print(f"{GREEN}SUCCESS: Build directory cleaned!{RESET}")
                log(f"Build directory cleaned successfully")
                return
            except Exception as e:
                is_last_attempt = (i == max_retries - 1)
                if is_last_attempt:
                    print(f"{RED}ERROR: Failed to clean build directory after {max_retries} attempts: {e}{RESET}")
                    print(f"{YELLOW}Please manually close any programs using the build folder.{RESET}")
                    log(f"Failed to clean build directory: {e}")
                else:
                    print(f"{YELLOW}Clean failed, retrying in 1s... ({i+1}/{max_retries}){RESET}")
                    time.sleep(1)
    else:
        print(f"{YELLOW}Build directory does not exist.{RESET}")
        log("Build directory did not exist")

def build_configuration(cmake_path, config, compiler_name):
    clear_screen()
    print()
    print(f"{CYAN}================================================{RESET}")
    print(f"{CYAN}       Building {config} Configuration{RESET}")
    print(f"{CYAN}================================================{RESET}")
    print()

    log(f"Starting {config} Build")

    
    # Smart clean: only clean if generator changed
    generator_name = "Visual Studio 16 2019" # Default fallback
    use_ninja = False
    
    ninja_path = find_ninja_tool()
    if get_os_type() == "windows" and ninja_path:
        use_ninja = True
        generator_name = "Ninja"
    
    current_gen = get_current_generator()
    if current_gen and current_gen != generator_name:
        print(f"{YELLOW}Generator changed ({current_gen} -> {generator_name}). Cleaning build dir...{RESET}")
        clean_build_dir()
    elif not build_dir.exists():
        # Clean not needed, but ensure dir exists
        pass
    else:
        print(f"{GREEN}Incremental build detected (Generator: {generator_name}){RESET}")

    os_type = get_os_type()
    if os_type == "windows":
        if use_ninja:
            print(f"{GREEN}Using Ninja build system: {ninja_path}{RESET}")
            cmd_configure = f'"{cmake_path}" -B "{build_dir}" -G "Ninja" -DCMAKE_MAKE_PROGRAM="{ninja_path}" -DCMAKE_BUILD_TYPE={config} -S .'
            cmd_build = f'"{cmake_path}" --build "{build_dir}" --config {config}'
            
            # Source vcvars64.bat for Ninja
            vcvars = find_vcvars_bat()
            if vcvars:
                print(f"{BLUE}Sourcing vcvars: {vcvars}{RESET}")
                cmd_configure = f'"{vcvars}" && {cmd_configure}'
                cmd_build = f'"{vcvars}" && {cmd_build}'
            else:
                print(f"{YELLOW}WARNING: vcvars64.bat not found! Ninja build might fail.{RESET}")
        else:
            generator = "Visual Studio 16 2019"
            cmd_configure = f'"{cmake_path}" -B "{build_dir}" -G "{generator}" -A x64 -S .'
            cmd_build = f'"{cmake_path}" --build "{build_dir}" --config {config}'
    else:
        # For Linux/macOS use Unix Makefiles generator
        # Check for Ninja on Linux too
        if shutil.which("ninja"):
             generator = "Ninja"
             cmd_configure = f'cmake -B "{build_dir}" -G "{generator}" -DCMAKE_BUILD_TYPE={config}'
        else:
             generator = "Unix Makefiles"
             cmd_configure = f'cmake -B "{build_dir}" -G "{generator}" -DCMAKE_BUILD_TYPE={config}'
        cmd_build = f'cmake --build "{build_dir}" --config {config}'

    # Configure project
    print(f"{BLUE}Configuring project with {compiler_name}...{RESET}")
    ret = run_command(cmd_configure)
    if ret != 0:
        print()
        print(f"{RED}ERROR: Configuration failed!{RESET}")
        print("Please check your compiler installation.")
        log("Configuration failed")
        pause()
        return False

    # Build configuration
    print()
    print(f"{BLUE}Building {config} configuration...{RESET}")
    ret = run_command(cmd_build)
    if ret != 0:
        print()
        print(f"{RED}ERROR: Build failed!{RESET}")
        log("Build failed")
        pause()
        return False

    print()
    print(f"{GREEN}SUCCESS: {config} build completed!{RESET}")
    print()
    print(f"{WHITE}Executables created:{RESET}")
    
    ext = ".exe" if os_type == "windows" else ""
    
    # Check both config-specific and root bin folders (Ninja vs VS)
    bin_config = build_dir / 'bin' / config
    bin_root = build_dir / 'bin'
    
    # helper to find exe
    def find_exe(name):
        p1 = bin_config / f"{name}{ext}"
        if p1.exists(): return p1
        p2 = bin_root / f"{name}{ext}"
        if p2.exists(): return p2
        return p1 # default for display
        
    cgrunner_path = find_exe("cgrunner")
    test_path = find_exe("test_diagnostics")
    gui_path = find_exe("cgroot_gui")

    print(f"- {cgrunner_path}")
    print(f"- {test_path}")
    
    # Check for GUI executable (only if Qt6 was found)
    if gui_path.exists():
        print(f"- {gui_path} {GREEN}(GUI){RESET}")
    else:
        print(f"- {gui_path} {YELLOW}(not built - Qt6 may not be found){RESET}")
    
    log(f"{config} build completed successfully")
    pause()
    return True

def run_executables(configuration):
    clear_screen()
    print()
    print(f"{CYAN}================================================{RESET}")
    print(f"{CYAN}       Running {configuration} Executables{RESET}")
    print(f"{CYAN}================================================{RESET}")
    print()

    log(f"Running {configuration} Executables")

    os_type = get_os_type()
    ext = ".exe" if os_type == "windows" else ""

    # Helper to find exe in config dir or root bin (Ninja support)
    def get_exe_path(name):
        # Try config folder first (VS)
        p = build_dir / f"bin/{configuration}/{name}{ext}"
        if p.exists(): return p
        # Try root bin folder (Ninja)
        p = build_dir / f"bin/{name}{ext}"
        if p.exists(): return p
        # Default back to config for reporting missing
        return build_dir / f"bin/{configuration}/{name}{ext}"

    main_exec = get_exe_path("cgrunner")
    test_exec = get_exe_path("test_diagnostics")
    gui_exec = get_exe_path("cgroot_gui")

    executables_found = False

    if main_exec.exists():
        executables_found = True
        print(f"{GREEN}Running main executable (cgrunner)...{RESET}")
        print(f"{CYAN}========================================={RESET}")
        subprocess.run(str(main_exec))
        print()
    else:
        print(f"{YELLOW}WARNING: cgrunner executable not found (skipping){RESET}")
        print()

    if test_exec.exists():
        executables_found = True
        print(f"{GREEN}Running diagnostics executable (test_diagnostics)...{RESET}")
        print(f"{CYAN}========================================={RESET}")
        subprocess.run(str(test_exec))
        print()
    else:
        print(f"{YELLOW}WARNING: test_diagnostics executable not found (skipping){RESET}")
        print()

    # Launch Python GUI (cgroot_gui is no longer a C++ executable but handled via Python)
    print(f"{GREEN}Preparing Python GUI (PyQt6)...{RESET}")
    print(f"{CYAN}========================================={RESET}")
    
    gui_script = project_root / "src" / "gui_py" / "main.py"
    if gui_script.exists():
        print(f"{GREEN}Launching GUI script: {gui_script}{RESET}")
        print(f"{YELLOW}Note: GUI will open in a separate window. Close the window to continue.{RESET}")
        print()
        try:
            # Run Python GUI
            if get_os_type() == "windows":
                # Use pythonw or python to run script
                result = subprocess.run(
                    f'python "{gui_script}"',
                    shell=True,
                    timeout=None # GUI might run for a while
                )
            else:
                result = subprocess.run(
                    ["python3", str(gui_script)],
                    timeout=None
                )
            
            if result.returncode != 0:
                print(f"{RED}GUI exited with error code {result.returncode}{RESET}")
        except KeyboardInterrupt:
             print(f"{YELLOW}GUI execution interrupted.{RESET}")
        except Exception as e:
            print(f"{RED}Error running GUI: {e}{RESET}")
    else:
        print(f"{RED}ERROR: Python GUI script not found at {gui_script}{RESET}")
    print()

    if not executables_found:
        print(f"{RED}ERROR: No executables found!{RESET}")
        print(f"Please build the {configuration} configuration first.")
        log("No executables found")
        pause()
        return

    print(f"{GREEN}SUCCESS: All available {configuration} executables completed!{RESET}")
    log(f"{configuration} executables completed successfully")
    pause()

def build_and_run(cmake_path, config, compiler_name):
    clear_screen()
    print()
    print(f"{CYAN}================================================{RESET}")
    print(f"{CYAN}     Building and Running {config} Configuration{RESET}")
    print(f"{CYAN}================================================{RESET}")
    print()

    log(f"Starting Build and Run {config}")

    # clean_build_dir()

    
    # Smart clean logic
    generator_name = "Visual Studio 16 2019"
    use_ninja = False
    
    ninja_path = find_ninja_tool()
    if get_os_type() == "windows" and ninja_path:
        use_ninja = True
        generator_name = "Ninja"

    current_gen = get_current_generator()
    if current_gen and current_gen != generator_name:
         print(f"{YELLOW}Generator changed. Cleaning build dir...{RESET}")
         clean_build_dir()

    os_type = get_os_type()
    if os_type == "windows":
        if use_ninja:
            print(f"{GREEN}Using Ninja build system: {ninja_path}{RESET}")
            cmd_configure = f'"{cmake_path}" -B "{build_dir}" -G "Ninja" -DCMAKE_MAKE_PROGRAM="{ninja_path}" -DCMAKE_BUILD_TYPE={config} -S .'
            
             # Source vcvars64.bat for Ninja
            vcvars = find_vcvars_bat()
            if vcvars:
                print(f"{BLUE}Sourcing vcvars: {vcvars}{RESET}")
                cmd_configure = f'"{vcvars}" && {cmd_configure}'
                
            # Note: build_and_run runs configure then build, but locally wraps calls. 
            # Wait, run_command takes shell=True so wrapping works.
            # But here cmd_build is passed to run_command later. So wrap it too.
            cmd_build = f'"{cmake_path}" --build "{build_dir}" --config {config}'
            if vcvars:
                 cmd_build = f'"{vcvars}" && {cmd_build}'

        else:
            generator = "Visual Studio 16 2019"
            cmd_configure = f'"{cmake_path}" -B "{build_dir}" -G "{generator}" -A x64 -S .'
            cmd_build = f'"{cmake_path}" --build "{build_dir}" --config {config}'
    else:
        if shutil.which("ninja"):
             generator = "Ninja"
             cmd_configure = f'cmake -B "{build_dir}" -G "{generator}" -DCMAKE_BUILD_TYPE={config}'
        else:
             generator = "Unix Makefiles"
             cmd_configure = f'cmake -B "{build_dir}" -G "{generator}" -DCMAKE_BUILD_TYPE={config}'
        cmd_build = f'cmake --build "{build_dir}" --config {config}'

    print(f"{BLUE}Configuring project with {compiler_name}...{RESET}")
    ret = run_command(cmd_configure)
    if ret != 0:
        print()
        print(f"{RED}ERROR: Configuration failed!{RESET}")
        log("Configuration failed")
        pause()
        return False

    print()
    print(f"{BLUE}Building {config} configuration...{RESET}")
    ret = run_command(cmd_build)
    if ret != 0:
        print()
        print(f"{RED}ERROR: Build failed!{RESET}")
        log("Build failed")
        pause()
        return False

    print()
    print(f"{GREEN}SUCCESS: {config} build completed!{RESET}")
    print(f"{GREEN}Running executables...{RESET}")
    print(f"{CYAN}========================================={RESET}")

    run_executables(config)
    log(f"Build and run {config} completed successfully")
    return True

def show_status(compiler_name):
    clear_screen()
    print()
    print(f"{CYAN}================================================{RESET}")
    print(f"{CYAN}            Project Status{RESET}")
    print(f"{CYAN}================================================{RESET}")
    print()

    print(f"{WHITE}Project Directory: {os.getcwd()}{RESET}")
    print(f"{WHITE}Compiler: {compiler_name}{RESET}")
    print(f"{WHITE}Build Directory: {build_dir}{RESET}")
    print()

    print(f"{BLUE}Build Directory Status:{RESET}")
    if build_dir.exists():
        print(f"{GREEN}[EXISTS]{RESET} {build_dir}")
        os_type = get_os_type()
        ext = ".exe" if os_type == "windows" else ""
        for cfg in ['Debug', 'Release']:
            print(f"  Configuration: {cfg}")
            
            # Helper to check paths
            def check_exe(name):
                # VS style
                p1 = build_dir / f"bin/{cfg}/{name}{ext}"
                # Ninja style (usually no config subdir for single-config generator, but we'll check root bin)
                p2 = build_dir / f"bin/{name}{ext}"
                
                if p1.exists(): return p1, True
                # Only check p2 if we are strictly in that mode? confusing for status.
                # Just check if p1 exists. If not, check p2 ONLY if cfg matches cmake_build_type?
                # For simplicity, check both and report found one.
                if p2.exists(): return p2, True
                return p1, False

            for name in ["cgrunner", "test_diagnostics", "cgroot_gui"]:
                path, exists = check_exe(name)
                status_str = "EXISTS" if exists else "MISSING"
                color = GREEN if exists else (YELLOW if "gui" in name else RED)
                print(f"    {color}[{status_str}]{RESET} {name}")
    else:
        print(f"{RED}[MISSING]{RESET} {build_dir}")

    print()
    print(f"{BLUE}Source Files:{RESET}")
    for f in ["src/main.cpp", "tests/test_diagnostics.cpp"]:
        file_path = project_root / f
        if file_path.exists():
            print(f"{GREEN}[EXISTS]{RESET} {f}")
        else:
            print(f"{RED}[MISSING]{RESET} {f}")

    print()
    print(f"{BLUE}CMake Configuration:{RESET}")
    cmake_file = project_root / "CMakeLists.txt"
    if cmake_file.exists():
        print(f"{GREEN}[EXISTS]{RESET} CMakeLists.txt")
    else:
        print(f"{RED}[MISSING]{RESET} CMakeLists.txt")

    print()
    print(f"{BLUE}Test Files:{RESET}")
    for f in ["tests/test_tensor.cpp", "tests/test_autograd.cpp"]:
        file_path = project_root / f
        if file_path.exists():
            print(f"{GREEN}[EXISTS]{RESET} {f}")
        else:
            print(f"{RED}[MISSING]{RESET} {f}")

    print()
    pause()

def open_vs_solution():
    clear_screen()
    print()
    print(f"{CYAN}================================================{RESET}")
    print(f"{CYAN}        Opening Visual Studio Solution{RESET}")
    print(f"{CYAN}================================================{RESET}")
    print()

    # Solution file is in build directory or project root
    sln_path = build_dir / "CGROOT++.sln"
    if not sln_path.exists():
        sln_path = project_root / "CGROOT++.sln"
    
    if not sln_path.exists():
        print(f"{RED}ERROR: Visual Studio solution not found!{RESET}")
        print("Please build the project first (Options 1 or 2).")
        log("VS solution not found")
        pause()
        return

    print(f"{BLUE}Opening Visual Studio solution...{RESET}")
    if get_os_type() == "windows":
        subprocess.Popen(f'start "" "{sln_path}"', shell=True)
    else:
        # On Linux, open with default app (code as example or xdg-open)
        subprocess.Popen(["xdg-open", str(sln_path)])
    print(f"{GREEN}SUCCESS: Visual Studio solution opened!{RESET}")
    log("VS solution opened")
    pause()

def run_tests():
    clear_screen()
    print()
    print(f"{CYAN}================================================{RESET}")
    print(f"{CYAN}              Running Tests{RESET}")
    print(f"{CYAN}================================================{RESET}")
    print()

    log("Running tests")

    os_type = get_os_type()
    ext = ".exe" if os_type == "windows" else ""

    test_found = False

    debug_main = build_dir / f"bin/Debug/cgrunner{ext}"
    debug_test = build_dir / f"bin/Debug/test_diagnostics{ext}"
    debug_gui = build_dir / f"bin/Debug/cgroot_gui{ext}"

    if debug_main.exists():
        test_found = True
        print(f"{GREEN}Running main executable tests...{RESET}")
        print(f"{CYAN}========================================={RESET}")
        subprocess.run(str(debug_main))
        print()

    if debug_test.exists():
        test_found = True
        print(f"{GREEN}Running diagnostics test executable...{RESET}")
        print(f"{CYAN}========================================={RESET}")
        subprocess.run(str(debug_test))
        print()

    if debug_gui.exists():
        test_found = True
        print(f"{GREEN}Preparing GUI executable...{RESET}")
        print(f"{CYAN}========================================={RESET}")
        
        # Deploy Qt DLLs for the GUI executable
        # This will copy the exe to a separate folder and deploy Qt DLLs
        deployed_gui_exec = deploy_qt_dlls(debug_gui, "Debug")
        
        if not deployed_gui_exec or not deployed_gui_exec.exists():
            print(f"{RED}ERROR: Failed to deploy GUI executable{RESET}")
            print(f"{YELLOW}Skipping GUI launch.{RESET}")
            print()
        else:
            print(f"{GREEN}Launching GUI executable...{RESET}")
            print(f"{YELLOW}Note: GUI will open in a separate window. Close the window to continue.{RESET}")
            print()
            try:
                # Use the deployed executable path (handles spaces correctly)
                if get_os_type() == "windows":
                    # Use quotes around path to handle spaces
                    result = subprocess.run(
                        f'"{deployed_gui_exec}"',
                        shell=True,
                        timeout=300
                    )
                else:
                    result = subprocess.run(
                        str(deployed_gui_exec),
                        timeout=300
                    )
                if result.returncode != 0:
                    print(f"{RED}GUI exited with error code {result.returncode}{RESET}")
                    print(f"{YELLOW}If the GUI window didn't appear, check for runtime errors.{RESET}")
            except subprocess.TimeoutExpired:
                print(f"{YELLOW}GUI was closed or timed out.{RESET}")
            except Exception as e:
                print(f"{RED}Error running GUI: {e}{RESET}")
                print(f"{YELLOW}Try running the GUI directly: {deployed_gui_exec}{RESET}")
            print()

    if not test_found:
        print(f"{RED}ERROR: No test executables found!{RESET}")
        print("Please build the Debug configuration first (Option 1).")
        log("No test executables found")
        pause()
        return

    print(f"{GREEN}SUCCESS: All tests completed!{RESET}")
    log("Tests completed successfully")
    pause()

def generate_docs():
    clear_screen()
    print()
    print(f"{CYAN}================================================{RESET}")
    print(f"{CYAN}         Generating Documentation{RESET}")
    print(f"{CYAN}================================================{RESET}")
    print()

    log("Generating documentation")

    docs_path = Path("docs")
    if not docs_path.exists():
        docs_path.mkdir()

    readme_path = docs_path / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# CGROOT++ Documentation\n\n")
        f.write(f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Project Structure\n\n")
        f.write("### Core Components\n")
        f.write("- **tensor.h/cpp**: Core tensor operations\n")
        f.write("- **parameter.h**: Parameter management\n")
        f.write("- **shape.h**: Shape handling\n\n")
        f.write("### Neural Network Components\n")
        f.write("- **linear.h**: Linear layer implementation\n")
        f.write("- **relu.h**: ReLU activation function\n")
        f.write("- **sigmoid.h**: Sigmoid activation function\n")
        f.write("- **conv2d.h**: 2D Convolution layer\n")
        f.write("- **sequential.h**: Sequential model container\n\n")
        f.write("### Loss Functions\n")
        f.write("- **mse_loss.h**: Mean Squared Error loss\n")
        f.write("- **cross_entropy_loss.h**: Cross Entropy loss\n\n")
        f.write("### Optimizers\n")
        f.write("- **sgd.h**: Stochastic Gradient Descent\n")
        f.write("- **adam.h**: Adam optimizer\n\n")
        f.write("### Math Operations\n")
        f.write("- **cpu_kernels.h/cpp**: CPU-based mathematical operations\n\n")
        f.write("### Autograd System\n")
        f.write("- **graph.h/cpp**: Computational graph\n")
        f.write("- **op_nodes.h/cpp**: Operation nodes\n")
    print(f"{GREEN}SUCCESS: Documentation generated in docs/ directory!{RESET}")
    log("Documentation generated successfully")
    pause()

def show_log():
    clear_screen()
    print()
    print(f"{CYAN}================================================{RESET}")
    print(f"{CYAN}              Build Log{RESET}")
    print(f"{CYAN}================================================{RESET}")
    print()

    if log_file.exists():
        print(f"{BLUE}Contents of {log_file}:{RESET}")
        print(f"{CYAN}========================================={RESET}")
        with open(log_file, "r", encoding="utf-8") as f:
            print(f.read())
        print(f"{CYAN}========================================={RESET}")
    else:
        print(f"{YELLOW}No build log found.{RESET}")
    print()
    pause()

def main_menu(cmake_cmd, compiler_name):

    clear_screen()

    if "--clean" in sys.argv:
        clean_build_dir()
        if "--build" in sys.argv:
            build_and_run(cmake_cmd, "Release", compiler_name)
        return False

    if "--build" in sys.argv:
        build_and_run(cmake_cmd, "Release", compiler_name)
        return False


    while True:
        print()
        print(f"{CYAN}================================================{RESET}")
        print(f"{CYAN}           {project_name} Project Manager{RESET}")
        print(f"{CYAN}================================================{RESET}")
        print()
        print(f"{BLUE}Compiler in use: {GREEN}{compiler_name}{RESET}")
        print(f"{BLUE}Build directory: {GREEN}{build_dir}{RESET}")
        print()
        print(f"{WHITE}Choose an option:{RESET}")
        print()
        print(f"{GREEN}1.{RESET} Build Debug Configuration")
        print(f"{GREEN}2.{RESET} Build Release Configuration")
        print(f"{GREEN}3.{RESET} Run Debug Executables")
        print(f"{GREEN}4.{RESET} Run Release Executables")
        print(f"{GREEN}5.{RESET} Build and Run Debug")
        print(f"{GREEN}6.{RESET} Build and Run Release")
        print(f"{GREEN}7.{RESET} Clean Build Directory")
        print(f"{GREEN}8.{RESET} Show Project Status")
        print(f"{GREEN}9.{RESET} Open Visual Studio Solution")
        print(f"{GREEN}10.{RESET} Run Tests")
        print(f"{GREEN}11.{RESET} Generate Documentation")
        print(f"{GREEN}12.{RESET} Show Build Log")
        print(f"{GREEN}13.{RESET} Change Compiler")
        print(f"{RED}0.{RESET} Exit")
        print()
        choice = input(f"{YELLOW}Enter your choice (0-13): {RESET}").strip()

        if choice == "1":
            build_configuration(cmake_cmd, "Debug", compiler_name)
        elif choice == "2":
            build_configuration(cmake_cmd, "Release", compiler_name)
        elif choice == "3":
            run_executables("Debug")
        elif choice == "4":
            run_executables("Release")
        elif choice == "5":
            build_and_run(cmake_cmd, "Debug", compiler_name)
        elif choice == "6":
            build_and_run(cmake_cmd, "Release", compiler_name)
        elif choice == "7":
            clean_build_dir()
            pause()
        elif choice == "8":
            show_status(compiler_name)
        elif choice == "9":
            open_vs_solution()
        elif choice == "10":
            run_tests()
        elif choice == "11":
            generate_docs()
        elif choice == "12":
            show_log()
        elif choice == "13":
            return True  # change compiler
        elif choice == "0":
            clear_screen()
            print()
            print(f"{GREEN}Thank you for using {project_name} Project Manager!{RESET}")
            log("Manager closed")
            print()
            sys.exit(0)
        else:
            print(f"{RED}Invalid choice! Please try again.{RESET}")
            pause()

def run_tests_only(config):
    # Only run test_diagnostics
    os_type = get_os_type()
    ext = ".exe" if os_type == "windows" else ""
    
    # Try to find test executable
    candidates = [
        build_dir / "bin" / config / f"test_diagnostics{ext}",
        build_dir / "bin" / f"test_diagnostics{ext}"
    ]
    
    test_exec = None
    for c in candidates:
        if c.exists():
            test_exec = c
            break
            
    if test_exec:
        print(f"{GREEN}Running diagnostics executable (test_diagnostics)...{RESET}")
        print(f"{CYAN}========================================={RESET}")
        subprocess.run(str(test_exec))
    else:
         print(f"{RED}ERROR: test_diagnostics executable not found.{RESET}")
         print(f"Please build the project first using --build.")

def main():
    while True:
        available = detect_compilers()
        compiler_name, cmake_cmd = choose_compiler(available)
        print()
        print(f"{GREEN}Selected compiler: {compiler_name}{RESET}")
        print(f"{BLUE}Path: {cmake_cmd}{RESET}")

        make_tool = detect_make_tool()
        if make_tool:
            print(f"{GREEN}Using make tool: {make_tool}{RESET}")
        else:
            print(f"{YELLOW}WARNING:{RESET} No make tool found (make.exe, mingw32-make.exe, nmake.exe)")
            print("You may need to install MinGW, MSYS2, or AVR-GCC tools.")
        log(f"Compiler: {compiler_name}")
        log(f"Path: {cmake_cmd}")
        pause()

        change_compiler = main_menu(cmake_cmd, compiler_name)
        if not change_compiler:
            break

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_tests_only("Release")
    else:
        main()
