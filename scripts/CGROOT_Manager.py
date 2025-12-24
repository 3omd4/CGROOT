import os
import sys
import subprocess
import datetime
from pathlib import Path
import platform
import shutil
import stat
import errno
import time

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

# ==========================================
# AUTO-DETECTION HELPERS (No Hardcoded Paths)
# ==========================================

def find_vswhere():
    """Locate vswhere.exe for Visual Studio detection on Windows."""
    if get_os_type() != "windows": return None
    
    # Check standard location
    roots = [os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")]
    for root in roots:
        if root:
            path = Path(root) / "Microsoft Visual Studio/Installer/vswhere.exe"
            if path.exists(): return path
    return None

def get_vs_installations():
    """
    Get list of Visual Studio installation paths using vswhere.
    Returns list of Path objects.
    """
    vswhere = find_vswhere()
    installations = []
    
    if vswhere:
        try:
            # Find VS with VC++ tools
            output = subprocess.check_output([
                str(vswhere), "-latest", "-products", "*", 
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", 
                "-property", "installationPath"
            ], encoding="utf-8", stderr=subprocess.DEVNULL)
            
            for line in output.splitlines():
                if line.strip():
                    installations.append(Path(line.strip()))
        except:
            pass
            
    # Fallback: Check common paths if vswhere fails or returns nothing
    if not installations:
        common_roots = [
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019"),
            Path(r"C:\Program Files\Microsoft Visual Studio\2022")
        ]
        for root in common_roots:
            if root.exists():
                for edition in ["Community", "Professional", "Enterprise", "BuildTools"]:
                    p = root / edition
                    if p.exists():
                        installations.append(p)
                        
    return installations

def find_vcvars_bat():
    """Auto-detect vcvars64.bat."""
    if get_os_type() != "windows": return None
    
    # 1. Check via detected VS installations
    installations = get_vs_installations()
    for install_dir in installations:
        vcvars = install_dir / "VC/Auxiliary/Build/vcvars64.bat"
        if vcvars.exists(): return str(vcvars)
        
    return None

def find_ninja_tool():
    """Auto-detect ninja.exe."""
    # 1. Check PATH
    if shutil.which("ninja"): return "ninja"
    
    if get_os_type() == "windows":
        # 2. Check VS Extensions
        installations = get_vs_installations()
        for install_dir in installations:
            ninja = install_dir / "Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"
            if ninja.exists(): return str(ninja)
            
        # 3. Check standalone CMake location
        cmake_ninja = Path(r"C:\Program Files\CMake\bin\ninja.exe")
        if cmake_ninja.exists(): return str(cmake_ninja)
        
    return None

def detect_compilers():
    """Auto-detect available CMake instances."""
    os_type = get_os_type()
    if os_type != "windows":
        if shutil.which("cmake"):
            return {1: ("System CMake", "cmake")}
        else:
            print(f"{RED}ERROR: 'cmake' not found in PATH.{RESET}")
            sys.exit(1)

    available = {}
    idx = 1
    
    print()
    print(f"{CYAN}================================================{RESET}")
    print(f"{CYAN}           {project_name} Project Manager{RESET}")
    print(f"{CYAN}================================================{RESET}")
    
    # 1. System CMake (PATH)
    sys_cmake = shutil.which("cmake")
    if sys_cmake:
        available[idx] = ("System CMake (PATH)", sys_cmake)
        print(f"[{idx}] {GREEN}System CMake{RESET} found in PATH")
        idx += 1
        
    # 2. Visual Studio CMake
    installations = get_vs_installations()
    for install_dir in installations:
        cmake = install_dir / "Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
        if cmake.exists():
             name = f"VS CMake ({install_dir.name})"
             # Avoid duplicate if PATH point to same
             if sys_cmake and Path(sys_cmake).resolve() == cmake.resolve():
                 continue 
             available[idx] = (name, str(cmake))
             print(f"[{idx}] {GREEN}{name}{RESET}")
             idx += 1
             
    # 3. Standard Program Files CMake
    std_cmake = Path(r"C:\Program Files\CMake\bin\cmake.exe")
    if std_cmake.exists():
        if not sys_cmake or Path(sys_cmake).resolve() != std_cmake.resolve():
             available[idx] = ("Standard CMake", str(std_cmake))
             print(f"[{idx}] {GREEN}Standard CMake{RESET}")
             idx += 1
             
    # 4. Detect others (MinGW/MSYS) via standard paths if needed
    extra_paths = [
        ("MinGW CMake", r"C:\mingw64\bin\cmake.exe"),
        ("MSYS2 UCRT64 CMake", r"C:\msys64\ucrt64\bin\cmake.exe"),
        ("MSYS2 MinGW64 CMake", r"C:\msys64\mingw64\bin\cmake.exe"),
    ]
    for name, p in extra_paths:
        path = Path(p)
        if path.exists():
            available[idx] = (name, str(path))
            print(f"[{idx}] {GREEN}{name}{RESET}")
            idx += 1

    print()
    if not available:
        print(f"{RED}ERROR: No supported CMake found!{RESET}")
        print(f"{CYAN}Please install Visual Studio Build Tools, CMake, or MinGW.{RESET}")
        input("Press Enter to exit...")
        sys.exit(1)
        
    return available

def detect_make_tool():
    """Auto-detect make tool."""
    if get_os_type() != "windows":
        return "make" if shutil.which("make") else None
        
    # Windows: Check common names in PATH
    for tool in ["make", "mingw32-make", "nmake"]:
        if shutil.which(tool):
            return tool
            
    # Check VS nmake
    installations = get_vs_installations()
    for install_dir in installations:
        nmake = install_dir / "VC/Tools/MSVC"
        if nmake.exists():
            # Find version dir
            for version_dir in nmake.iterdir():
                nmake_exe = version_dir / "bin/Hostx64/x64/nmake.exe"
                if nmake_exe.exists(): return str(nmake_exe)
                
    # Check MinGW/MSYS
    common_paths = [
        Path(r"C:\mingw64\bin\mingw32-make.exe"),
        Path(r"C:\msys64\usr\bin\make.exe"),
        Path(r"C:\avr-gcc\bin\make.exe"),
    ]
    for p in common_paths:
        if p.exists(): return str(p)
        
    return None

def find_windeployqt():
    """
    Find windeployqt.exe using PATH and smart search.
    """
    if get_os_type() != "windows": return None
    
    # 1. Check PATH
    if shutil.which("windeployqt"):
        return Path(shutil.which("windeployqt"))
    
    # 2. Check CMAKE_PREFIX_PATH
    cmake_prefix = os.getenv("CMAKE_PREFIX_PATH", "")
    if cmake_prefix:
        for path in cmake_prefix.split(os.pathsep):
            wd = Path(path) / "bin/windeployqt.exe"
            if wd.exists(): return wd

    # 3. Search standard Qt locations (glob)
    roots = [
        Path(r"C:\Qt"),
        Path.home() / "Qt"
    ]
    for root in roots:
        if root.exists():
            # Look for version numbers (6.*)
            for ver_dir in root.glob("6.*"):
                if ver_dir.is_dir():
                    # Look for msvc/mingw compilers
                    for compiler_dir in ver_dir.iterdir():
                        wd = compiler_dir / "bin/windeployqt.exe"
                        if wd.exists(): return wd
                        
    return None

def get_cmake_and_make_paths():
    # Legacy wrapper used by original logic (kept for compatibility if needed, but detect_compilers replaces it)
    return get_os_type()

# ==========================================
# END AUTO-DETECTION
# ==========================================

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
            # Also check CMAKE_GENERATOR val
            f.seek(0)
            for line in f:
                 if line.startswith("CMAKE_GENERATOR:STRING="):
                     return line.split("=")[1].strip()
    except:
        pass
    return None

def deploy_qt_dlls(gui_executable, configuration):
    """
    Deploy Qt DLLs for the GUI executable using windeployqt.
    """
    if get_os_type() != "windows":
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
    
    # Create separate folder for GUI app
    gui_app_dir = build_dir / "bin" / configuration / "GuiApp"
    gui_app_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy GUI executable
    deployed_executable = gui_app_dir / gui_executable.name
    try:
        shutil.copy2(gui_executable, deployed_executable)
    except Exception as e:
        print(f"{RED}ERROR: Failed to copy GUI executable: {e}{RESET}")
        return None
    
    print(f"{BLUE}Deploying Qt DLLs...{RESET}")
    windeployqt_args = [
        str(windeployqt),
        "--compiler-runtime",
        "--force",
    ]
    
    if configuration.lower() == "release":
        windeployqt_args.append("--release") # Note: windeployqt --release might strictly look for release libs
    else:
        windeployqt_args.append("--debug")
    
    windeployqt_args.append(str(deployed_executable))
    
    try:
        result = subprocess.run(
            windeployqt_args,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"{GREEN}Successfully deployed Qt DLLs.{RESET}")
        else:
            print(f"{RED}windeployqt failed (Exit {result.returncode}){RESET}")
            if result.stderr: print(result.stderr)
            
        return deployed_executable
    except Exception as e:
        print(f"{RED}ERROR: Failed to run windeployqt: {e}{RESET}")
        return deployed_executable

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
            print(f"{RED}Invalid choice.{RESET}")
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

def remove_readonly(func, path, exc):
    excvalue = exc[1]
    if func in (os.rmdir, os.remove, os.unlink) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise

def kill_zombie_processes():
    if get_os_type() != "windows": return

    targets = [
        "cmake.exe", "ninja.exe", "msbuild.exe", "make.exe",
        "cgrunner.exe", "test_diagnostics.exe", "cgroot_gui.exe"
    ]

    print(f"{YELLOW}Checking for zombie processes...{RESET}")
    for target in targets:
        try:
            subprocess.run(f"taskkill /F /IM {target}", shell=True, 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except: pass

    try:
        subprocess.run('taskkill /F /FI "WINDOWTITLE eq CGROOT++ Neural Network Trainer"', shell=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except: pass

def clean_build_dir():
    kill_zombie_processes()
    print(f"{YELLOW}Cleaning build directory...{RESET}")

    if build_dir.exists() and build_dir.is_dir():
        for i in range(3):
            try:
                shutil.rmtree(build_dir, onerror=remove_readonly)
                print(f"{GREEN}SUCCESS: Build directory cleaned!{RESET}")
                log(f"Build directory cleaned successfully")
                break
            except Exception as e:
                if i == 2:
                    print(f"{RED}ERROR: Failed to clean build directory: {e}{RESET}")
                else:
                    time.sleep(1)
    else:
        print(f"{YELLOW}Build directory does not exist.{RESET}")

    print(f"{YELLOW}Cleaning __pycache__ directories...{RESET}")
    try:
        for pycache in project_root.rglob("__pycache__"):
            try:
                if pycache.is_dir():
                    shutil.rmtree(pycache, onerror=remove_readonly)
            except: pass
    except: pass
    pause()

def build_configuration(cmake_path, config, compiler_name):
    clear_screen()
    print()
    print(f"{CYAN}================================================{RESET}")
    print(f"{CYAN}       Building {config} Configuration{RESET}")
    print(f"{CYAN}================================================{RESET}")
    print()

    log(f"Starting {config} Build")
    
    # Auto-detect generator
    generator_name = "Visual Studio 16 2019" # Fallback
    use_ninja = False
    
    ninja_path = find_ninja_tool()
    if get_os_type() == "windows" and ninja_path:
        use_ninja = True
        generator_name = "Ninja"
    
    current_gen = get_current_generator()
    if current_gen and current_gen != generator_name:
        print(f"{YELLOW}Generator changed ({current_gen} -> {generator_name}). Cleaning build dir...{RESET}")
        clean_build_dir()

    os_type = get_os_type()
    if os_type == "windows":
        if use_ninja:
            print(f"{GREEN}Using Ninja build system: {ninja_path}{RESET}")
            cmd_configure = f'"{cmake_path}" -B "{build_dir}" -G "Ninja" -DCMAKE_MAKE_PROGRAM="{ninja_path}" -DCMAKE_BUILD_TYPE={config} -S .'
            cmd_build = f'"{cmake_path}" --build "{build_dir}" --config {config}'
            
            # Auto-source vcvars
            vcvars = find_vcvars_bat()
            if vcvars:
                print(f"{BLUE}Sourcing vcvars: {vcvars}{RESET}")
                cmd_configure = f'"{vcvars}" && {cmd_configure}'
                cmd_build = f'"{vcvars}" && {cmd_build}'
            else:
                print(f"{YELLOW}WARNING: vcvars64.bat not found (Ninja build may fail){RESET}")
        else:
            # Default VS Generator (try to detect 2022 vs 2019)
            generator = "Visual Studio 16 2019"
            # If 2022 is installed, use 17 2022?
            # We can deduce from compiler_name or vswhere.
            if "2022" in compiler_name or "2022" in str(cmake_path):
                 generator = "Visual Studio 17 2022"
            
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

    print(f"{BLUE}Configuring project...{RESET}")
    ret = run_command(cmd_configure)
    if ret != 0:
        print(f"{RED}ERROR: Configuration failed!{RESET}")
        log("Configuration failed")
        pause()
        return False

    print(f"{BLUE}Building...{RESET}")
    ret = run_command(cmd_build)
    if ret != 0:
        print(f"{RED}ERROR: Build failed!{RESET}")
        log("Build failed")
        pause()
        return False

    print(f"{GREEN}SUCCESS: {config} build completed!{RESET}")
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

    def get_exe_path(name):
        p = build_dir / f"bin/{configuration}/{name}{ext}"
        if p.exists(): return p
        p = build_dir / f"bin/{name}{ext}" # Ninja fallback
        if p.exists(): return p
        return build_dir / f"bin/{configuration}/{name}{ext}"

    main_exec = get_exe_path("cgrunner")
    test_exec = get_exe_path("test_diagnostics")

    executables_found = False

    if main_exec.exists():
        executables_found = True
        print(f"{GREEN}Running main executable (cgrunner)...{RESET}")
        print(f"{CYAN}========================================={RESET}")
        subprocess.run(str(main_exec))
        print()
    else:
        print(f"{YELLOW}WARNING: cgrunner not found{RESET}")

    if test_exec.exists():
        executables_found = True
        print(f"{GREEN}Running diagnostics...{RESET}")
        print(f"{CYAN}========================================={RESET}")
        subprocess.run(str(test_exec))
        print()
    else:
        print(f"{YELLOW}WARNING: test_diagnostics not found{RESET}")

    print(f"{GREEN}Preparing Python GUI...{RESET}")
    gui_script = project_root / "src" / "gui_py" / "main.py"
    if gui_script.exists():
        print(f"{GREEN}Launching GUI script: {gui_script}{RESET}")
        try:
            if get_os_type() == "windows":
                subprocess.run(f'python "{gui_script}"', shell=True)
            else:
                subprocess.run(["python3", str(gui_script)])
        except Exception as e:
            print(f"{RED}Error running GUI: {e}{RESET}")
    else:
        print(f"{RED}GUI script not found{RESET}")

    if not executables_found:
        print(f"{RED}ERROR: No C++ executables found!{RESET}")
        log("No executables found")
        
    pause()

def build_and_run(cmake_path, config, compiler_name):
    if build_configuration(cmake_path, config, compiler_name):
        run_executables(config)
        return True
    return False

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
    
    if build_dir.exists():
        print(f"{GREEN}[EXISTS]{RESET} {build_dir}")
    else:
        print(f"{RED}[MISSING]{RESET} {build_dir}")

    print()
    print(f"{BLUE}Executables:{RESET}")
    for exe in ["cgrunner.exe", "test_diagnostics.exe"]:
        found = False
        for loc in [build_dir/"bin"/"Release"/exe, build_dir/"bin"/"Debug"/exe, build_dir/"bin"/exe]:
            if loc.exists():
                print(f"  {GREEN}[FOUND]{RESET} {loc.relative_to(project_root)}")
                found = True
                break
        if not found:
             print(f"  {RED}[MISSING]{RESET} {exe}")

    pause()

def open_vs_solution():
    sln_path = build_dir / "CGROOT++.sln"
    if not sln_path.exists():
        sln_path = project_root / "CGROOT++.sln"
    
    if sln_path.exists():
        print(f"{BLUE}Opening {sln_path}...{RESET}")
        if get_os_type() == "windows":
            subprocess.Popen(f'start "" "{sln_path}"', shell=True)
        else:
            subprocess.Popen(["xdg-open", str(sln_path)])
    else:
        print(f"{RED}Solution file not found.{RESET}")
    pause()

def run_tests():
    run_executables("Debug") # Simplified test runner

def generate_docs():
    docs_path = Path("docs")
    docs_path.mkdir(exist_ok=True)
    with open(docs_path / "README.md", "w") as f:
        f.write("# Auto-Generated Docs\n")
    print(f"{GREEN}Docs generated in docs/{RESET}")
    pause()

def show_log():
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            print(f.read())
    else:
        print("No log.")
    pause()

def build_installer():
    print("Building installer...")
    if not shutil.which("pyinstaller"):
        print(f"{RED}PyInstaller not found.{RESET}")
        pause()
        return

    cmd = "pyinstaller CGROOT_Trainer.spec --noconfirm"
    run_command(cmd)
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
        print(f"Compiler: {GREEN}{compiler_name}{RESET}")
        print()
        print("1. Build Debug")
        print("2. Build Release")
        print("3. Run Debug")
        print("4. Run Release")
        print("5. Build & Run Debug")
        print("6. Build & Run Release")
        print("7. Clean")
        print("8. Status")
        print("9. Open VS Solution")
        print("10. Run Tests")
        print("11. Docs")
        print("12. Log")
        print("13. Change Compiler")
        print("14. Installer")
        print("0. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == "1": build_configuration(cmake_cmd, "Debug", compiler_name)
        elif choice == "2": build_configuration(cmake_cmd, "Release", compiler_name)
        elif choice == "3": run_executables("Debug")
        elif choice == "4": run_executables("Release")
        elif choice == "5": build_and_run(cmake_cmd, "Debug", compiler_name)
        elif choice == "6": build_and_run(cmake_cmd, "Release", compiler_name)
        elif choice == "7": clean_build_dir()
        elif choice == "8": show_status(compiler_name)
        elif choice == "9": open_vs_solution()
        elif choice == "10": run_tests()
        elif choice == "11": generate_docs()
        elif choice == "12": show_log()
        elif choice == "13": return True
        elif choice == "14": build_installer()
        elif choice == "0": sys.exit(0)

def main():
    while True:
        available = detect_compilers()
        compiler_name, cmake_cmd = choose_compiler(available)
        
        make_tool = detect_make_tool()
        if make_tool:
            print(f"{GREEN}Make tool: {make_tool}{RESET}")
            
        change = main_menu(cmake_cmd, compiler_name)
        if not change: break

if __name__ == "__main__":
    main()
