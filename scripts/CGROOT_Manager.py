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
build_dir = Path("build")
log_file = Path("build_log.txt")

# Windows-specific compiler CMake paths
username = os.getenv("USERNAME") or os.getenv("USER") or ""
cmake_paths = {
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

def clean_build_dir():
    if build_dir.exists() and build_dir.is_dir():
        print(f"{YELLOW}Cleaning build directory...{RESET}")
        shutil.rmtree(build_dir)
        print(f"{GREEN}SUCCESS: Build directory cleaned!{RESET}")
        log(f"Build directory cleaned successfully")
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

    if build_dir.exists():
        print(f"{YELLOW}Cleaning previous build...{RESET}")
        shutil.rmtree(build_dir)
        log(f"Cleaned previous build")

    os_type = get_os_type()
    if os_type == "windows":
        generator = "Visual Studio 16 2019"
        cmd_configure = f'"{cmake_path}" -B {build_dir} -G "{generator}"'
        cmd_build = f'"{cmake_path}" --build {build_dir} --config {config}'
    else:
        # For Linux/macOS use Unix Makefiles generator
        generator = "Unix Makefiles"
        cmd_configure = f'cmake -B {build_dir} -G "{generator}"'
        cmd_build = f'cmake --build {build_dir} --config {config}'

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
    print(f"- {build_dir}\\bin\\{config}{os.sep}cgrunner{ext}")
    print(f"- {build_dir}\\bin\\{config}{os.sep}simple_test{ext}")
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

    main_exec = build_dir / f"bin/{configuration}/cgrunner{ext}"
    test_exec = build_dir / f"bin/{configuration}/simple_test{ext}"

    if not main_exec.exists():
        print(f"{RED}ERROR: cgrunner executable not found!{RESET}")
        print(f"Please build the {configuration} configuration first.")
        log("cgrunner executable not found")
        pause()
        return

    if not test_exec.exists():
        print(f"{RED}ERROR: simple_test executable not found!{RESET}")
        print(f"Please build the {configuration} configuration first.")
        log("simple_test executable not found")
        pause()
        return

    print(f"{GREEN}Running main executable (cgrunner)...{RESET}")
    print(f"{CYAN}========================================={RESET}")
    subprocess.run(str(main_exec))
    print()
    print(f"{GREEN}Running example executable (simple_test)...{RESET}")
    print(f"{CYAN}========================================={RESET}")
    subprocess.run(str(test_exec))
    print()
    print(f"{GREEN}SUCCESS: All {configuration} executables completed!{RESET}")
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

    if build_dir.exists():
        print(f"{YELLOW}Cleaning previous build...{RESET}")
        shutil.rmtree(build_dir)
        log("Cleaned previous build")

    os_type = get_os_type()
    if os_type == "windows":
        generator = "Visual Studio 16 2019"
        cmd_configure = f'"{cmake_path}" -B {build_dir} -G "{generator}"'
        cmd_build = f'"{cmake_path}" --build {build_dir} --config {config}'
    else:
        generator = "Unix Makefiles"
        cmd_configure = f'cmake -B {build_dir} -G "{generator}"'
        cmd_build = f'cmake --build {build_dir} --config {config}'

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
            debug_cgrunner = build_dir / f"bin/{cfg}/cgrunner{ext}"
            debug_simple = build_dir / f"bin/{cfg}/simple_test{ext}"
            for exe, name in [(debug_cgrunner, f"{cfg}\\cgrunner{ext}"), (debug_simple, f"{cfg}\\simple_test{ext}")]:
                print(f"{GREEN if exe.exists() else RED}[{'EXISTS' if exe.exists() else 'MISSING'}]{RESET} {build_dir}\\bin\\{name}")
    else:
        print(f"{RED}[MISSING]{RESET} {build_dir}")

    print()
    print(f"{BLUE}Source Files:{RESET}")
    for f in ["src/main.cpp", "examples/simple_test.cpp"]:
        if Path(f).exists():
            print(f"{GREEN}[EXISTS]{RESET} {f}")
        else:
            print(f"{RED}[MISSING]{RESET} {f}")

    print()
    print(f"{BLUE}CMake Configuration:{RESET}")
    if Path("CMakeLists.txt").exists():
        print(f"{GREEN}[EXISTS]{RESET} CMakeLists.txt")
    else:
        print(f"{RED}[MISSING]{RESET} CMakeLists.txt")

    print()
    print(f"{BLUE}Test Files:{RESET}")
    for f in ["tests/test_tensor.cpp", "tests/test_autograd.cpp"]:
        if Path(f).exists():
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

    sln_path = build_dir / "CGROOT++.sln"
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
    debug_test = build_dir / f"bin/Debug/simple_test{ext}"

    if debug_main.exists():
        test_found = True
        print(f"{GREEN}Running main executable tests...{RESET}")
        print(f"{CYAN}========================================={RESET}")
        subprocess.run(str(debug_main))
        print()

    if debug_test.exists():
        test_found = True
        print(f"{GREEN}Running simple test executable...{RESET}")
        print(f"{CYAN}========================================={RESET}")
        subprocess.run(str(debug_test))
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
    while True:
        clear_screen()
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
    main()
