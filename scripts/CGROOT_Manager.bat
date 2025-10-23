@echo off

rem --- Ensure delayed expansion is enabled at top of script ---
setlocal enabledelayedexpansion
title CGROOT++ Project Manager

rem === Enable color output ===
rem Windows 10+ supports ANSI colors, set code page and enable virtual terminal processing
chcp 65001 >nul

rem Define ANSI color codes
rem Define the ESC character
for /F "delims=" %%A in ('echo prompt $E^| cmd') do set "ESC=%%A"

rem Define color variables using the ESC character
set "RED=%ESC%[31m"
set "GREEN=%ESC%[32m"
set "YELLOW=%ESC%[33m"
set "BLUE=%ESC%[34m"
set "MAGENTA=%ESC%[35m"
set "CYAN=%ESC%[36m"
set "WHITE=%ESC%[37m"
set "RESET=%ESC%[0m"

rem === Initialize variables ===
set "project_name=CGROOT++"
set "build_dir=build"
set "log_file=build_log.txt"

:INITIALIZE
cls
echo.
echo %CYAN%================================================%RESET%
echo %CYAN%           %project_name% Project Manager%RESET%
echo %CYAN%================================================%RESET%
echo.
echo %GREEN%Detecting available compilers...%RESET%
echo.

rem === Define cmake paths ===
set "vs2019_cmake=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "vs2022_cmake=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "vs2022_community_cmake=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "mingw_cmake=C:\mingw64\bin\cmake.exe"
set "msys2_cmake=C:\msys64\mingw64\bin\cmake.exe"
set "android_cmake=C:\Users\%USERNAME%\AppData\Local\Android\Sdk\cmake\3.31.6\bin\cmake.exe"
set "standalone_cmake=C:\Program Files\CMake\bin\cmake.exe"

rem === Possible make tools ===
set "avr_make=C:\avr-gcc\bin\make.exe"
set "msys_make=C:\msys64\usr\bin\make.exe"
set "mingw_make=C:\mingw64\bin\mingw32-make.exe"
set "vs_nmake=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\nmake.exe"

rem === Check availability ===
set idx=0
set available_list=

for %%A in (
    "%vs2019_cmake%"
    "%vs2022_cmake%"
    "%vs2022_community_cmake%"
    "%mingw_cmake%"
    "%msys2_cmake%"
    "%android_cmake%"
    "%standalone_cmake%"
) do (
    if exist %%~A (
        set /a idx+=1

        rem Use call set for indirect variable assignment with delayed expansion
        call set "compiler!idx!=%%~A"

        if "%%~A"=="%vs2019_cmake%" call set "compiler_name!idx!=Visual Studio 16 2019"
        if "%%~A"=="%vs2022_cmake%" call set "compiler_name!idx!=Visual Studio 17 2022 (Build Tools)"
        if "%%~A"=="%vs2022_community_cmake%" call set "compiler_name!idx!=Visual Studio 17 2022 (Community)"
        if "%%~A"=="%mingw_cmake%"  call set "compiler_name!idx!=MinGW (CMake)"
        if "%%~A"=="%msys2_cmake%"  call set "compiler_name!idx!=MSYS2 (CMake)"
        if "%%~A"=="%android_cmake%" call set "compiler_name!idx!=Android SDK CMake"
        if "%%~A"=="%standalone_cmake%" call set "compiler_name!idx!=Standalone CMake"

        call echo [!idx!] %%GREEN%%%%compiler_name!idx!%%%RESET% found at %%MAGENTA%%%%~A %%RESET%%
    )
)

echo.

if !idx! EQU 0 (
    echo %RED%ERROR: No supported compiler toolchains found!%RESET%
    echo %CYAN%Please install one of:
    echo   - Visual Studio Build Tools 2019 or 2022
    echo   - Visual Studio Community 2022
    echo   - MinGW or MSYS2
    echo   - Standalone CMake%RESET%
    pause
    exit /b 1
)

rem === Ask user for compiler choice ===
:CHOICE
set /p choice="%YELLOW%Enter compiler choice (1-!idx!): %RESET%"
echo.

rem Trim surrounding spaces (optional, simple trim)
for /f "tokens=* delims= " %%T in ("!choice!") do set "choice=%%T"
for /l %%I in (1,1,1) do if "!choice:~-1!"==" " set "choice=!choice:~0,-1" else goto :_trim_done
:_trim_done

rem Check empty
if "!choice!"=="" (
    echo %RED%No choice entered.%RESET%
    goto CHOICE
)

rem === Check if input contains any non-digit characters ===
for /f "delims=0123456789" %%A in ("!choice!") do (
    echo %RED%Invalid input. Enter digits only.%RESET%
    goto CHOICE
)

if !choice! LEQ 0 (
    echo %RED%Invalid choice. Enter a number between 1 and !idx!.%RESET%
    goto CHOICE
)

rem Convert to integer checks using delayed expansion
rem Check lower bound (must be >= 1)
if !choice! LSS 1 (
    echo %RED%Invalid choice. Enter a number between 1 and !idx!.%RESET%
    goto CHOICE
)

if !choice! GTR !idx! (
    echo %RED%Invalid choice. Enter a number between 1 and !idx!.%RESET%
    goto CHOICE
)

rem Set chosen compiler variables
call set "compiler_path=!compiler%choice%!"
call set "compiler_name=!compiler_name%choice%!"

echo.
echo %GREEN%Selected compiler: %compiler_name%%RESET%
echo %BLUE%Path: %compiler_path%%RESET%

rem === Auto-detect make tool ===
set "make_tool="

for %%M in ("%avr_make%" "%msys_make%" "%mingw_make%" "%vs_nmake%") do (
    if exist %%~M (
        set "make_tool=%%~M"
        goto found_make
    )
)

:found_make
if defined make_tool (
    echo %GREEN%Using make tool: !make_tool!%RESET%
) else (
    echo %YELLOW%WARNING:%RESET% No make tool found %CYAN%make.exe, mingw32-make.exe, nmake.exe%RESET%
    echo You may need to install MinGW, MSYS2, or AVR-GCC tools.
)
echo.

rem === Log initialization ===
echo %date% %time% - CGROOT++ Manager Started > "%log_file%"
echo Compiler: !compiler_name! >> "%log_file%"
echo Path: !compiler_path! >> "%log_file%"

echo.
pause

rem Now continue to your main menu or other logic...
goto :MAIN_MENU

:MAIN_MENU
cls
echo.
echo %CYAN%================================================%RESET%
echo %CYAN%           %project_name% Project Manager%RESET%
echo %CYAN%================================================%RESET%
echo.
echo %BLUE%Compiler in use: %GREEN%%compiler_name%%RESET%
echo %BLUE%Build directory: %GREEN%%build_dir%%RESET%
echo.
echo %WHITE%Choose an option:%RESET%
echo.
echo %GREEN%1.%RESET% Build Debug Configuration
echo %GREEN%2.%RESET% Build Release Configuration
echo %GREEN%3.%RESET% Run Debug Executables
echo %GREEN%4.%RESET% Run Release Executables
echo %GREEN%5.%RESET% Build and Run Debug
echo %GREEN%6.%RESET% Build and Run Release
echo %GREEN%7.%RESET% Clean Build Directory
echo %GREEN%8.%RESET% Show Project Status
echo %GREEN%9.%RESET% Open Visual Studio Solution
echo %GREEN%10.%RESET% Run Tests
echo %GREEN%11.%RESET% Generate Documentation
echo %GREEN%12.%RESET% Show Build Log
echo %GREEN%13.%RESET% Change Compiler
echo %RED%0.%RESET% Exit
echo.
set /p choice="%YELLOW%Enter your choice (0-13): %RESET%"

if "%choice%"=="1" goto BUILD_DEBUG
if "%choice%"=="2" goto BUILD_RELEASE
if "%choice%"=="3" goto RUN_DEBUG
if "%choice%"=="4" goto RUN_RELEASE
if "%choice%"=="5" goto BUILD_AND_RUN_DEBUG
if "%choice%"=="6" goto BUILD_AND_RUN_RELEASE
if "%choice%"=="7" goto CLEAN_BUILD
if "%choice%"=="8" goto SHOW_STATUS
if "%choice%"=="9" goto OPEN_VS_SOLUTION
if "%choice%"=="10" goto RUN_TESTS
if "%choice%"=="11" goto GENERATE_DOCS
if "%choice%"=="12" goto SHOW_LOG
if "%choice%"=="13" goto INITIALIZE
if "%choice%"=="0" goto EXIT
echo %RED%Invalid choice! Please try again.%RESET%
pause
goto MAIN_MENU

:BUILD_DEBUG
cls
echo.
echo %CYAN%================================================%RESET%
echo %CYAN%        Building Debug Configuration%RESET%
echo %CYAN%================================================%RESET%
echo.

echo %date% %time% - Starting Debug Build >> "%log_file%"

REM Clean previous build
if exist %build_dir% (
    echo %YELLOW%Cleaning previous build...%RESET%
    rmdir /s /q %build_dir%
    echo %date% %time% - Cleaned previous build >> "%log_file%"
)

REM Configure project
echo %BLUE%Configuring project with !compiler_name!...%RESET%
"%compiler_path%" -B %build_dir% -G "Visual Studio 16 2019" 2>&1 | tee -a "%log_file%"
if %errorlevel% neq 0 (
    echo.
    echo %RED%ERROR: Configuration failed!%RESET%
    echo Please check your compiler installation.
    echo %date% %time% - Configuration failed >> "%log_file%"
    pause
    goto MAIN_MENU
)

REM Build Debug configuration
echo.
echo %BLUE%Building Debug configuration...%RESET%
"%compiler_path%" --build %build_dir% --config Debug 2>&1 | tee -a "%log_file%"
if %errorlevel% neq 0 (
    echo.
    echo %RED%ERROR: Build failed!%RESET%
    echo %date% %time% - Build failed >> "%log_file%"
    pause
    goto MAIN_MENU
)

echo.
echo %GREEN%SUCCESS: Debug build completed!%RESET%
echo.
echo %WHITE%Executables created:%RESET%
echo - %build_dir%\bin\Debug\cgrunner.exe
echo - %build_dir%\bin\Debug\simple_test.exe
echo.
echo %date% %time% - Debug build completed successfully >> "%log_file%"
pause
goto MAIN_MENU

:BUILD_RELEASE
cls
echo.
echo %CYAN%================================================%RESET%
echo %CYAN%       Building Release Configuration%RESET%
echo %CYAN%================================================%RESET%
echo.

echo %date% %time% - Starting Release Build >> "%log_file%"

REM Clean previous build
if exist %build_dir% (
    echo %YELLOW%Cleaning previous build...%RESET%
    rmdir /s /q %build_dir%
    echo %date% %time% - Cleaned previous build >> "%log_file%"
)

REM Configure project
echo %BLUE%Configuring project with !compiler_name!...%RESET%
"%compiler_path%" -B %build_dir% -G "Visual Studio 16 2019" 2>&1 | tee -a "%log_file%"
if %errorlevel% neq 0 (
    echo.
    echo %RED%ERROR: Configuration failed!%RESET%
    echo Please check your compiler installation.
    echo %date% %time% - Configuration failed >> "%log_file%"
    pause
    goto MAIN_MENU
)

REM Build Release configuration
echo.
echo %BLUE%Building Release configuration...%RESET%
"%compiler_path%" --build %build_dir% --config Release 2>&1 | tee -a "%log_file%"
if %errorlevel% neq 0 (
    echo.
    echo %RED%ERROR: Build failed!%RESET%
    echo %date% %time% - Build failed >> "%log_file%"
    pause
    goto MAIN_MENU
)

echo.
echo %GREEN%SUCCESS: Release build completed!%RESET%
echo.
echo %WHITE%Executables created:%RESET%
echo - %build_dir%\bin\Release\cgrunner.exe
echo - %build_dir%\bin\Release\simple_test.exe
echo.
echo %date% %time% - Release build completed successfully >> "%log_file%"
pause
goto MAIN_MENU

:RUN_DEBUG
cls
echo.
echo %CYAN%================================================%RESET%
echo %CYAN%        Running Debug Executables%RESET%
echo %CYAN%================================================%RESET%
echo.

echo %date% %time% - Running Debug Executables >> "%log_file%"

REM Check if executables exist
if not exist "%build_dir%\bin\Debug\cgrunner.exe" (
    echo %RED%ERROR: cgrunner.exe not found!%RESET%
    echo Please build the Debug configuration first (Option 1).
    echo %date% %time% - cgrunner.exe not found >> "%log_file%"
    pause
    goto MAIN_MENU
)

if not exist "%build_dir%\bin\Debug\simple_test.exe" (
    echo %RED%ERROR: simple_test.exe not found!%RESET%
    echo Please build the Debug configuration first (Option 1).
    echo %date% %time% - simple_test.exe not found >> "%log_file%"
    pause
    goto MAIN_MENU
)

echo %GREEN%Running main executable (cgrunner.exe)...%RESET%
echo %CYAN%=========================================%RESET%
"%build_dir%\bin\Debug\cgrunner.exe"
echo.

echo %GREEN%Running example executable (simple_test.exe)...%RESET%
echo %CYAN%=========================================%RESET%
"%build_dir%\bin\Debug\simple_test.exe"
echo.

echo %GREEN%SUCCESS: All Debug executables completed!%RESET%
echo %date% %time% - Debug executables completed successfully >> "%log_file%"
pause
goto MAIN_MENU

:RUN_RELEASE
cls
echo.
echo %CYAN%================================================%RESET%
echo %CYAN%       Running Release Executables%RESET%
echo %CYAN%================================================%RESET%
echo.

echo %date% %time% - Running Release Executables >> "%log_file%"

REM Check if executables exist
if not exist "%build_dir%\bin\Release\cgrunner.exe" (
    echo %RED%ERROR: cgrunner.exe not found!%RESET%
    echo Please build the Release configuration first (Option 2).
    echo %date% %time% - cgrunner.exe not found >> "%log_file%"
    pause
    goto MAIN_MENU
)

if not exist "%build_dir%\bin\Release\simple_test.exe" (
    echo %RED%ERROR: simple_test.exe not found!%RESET%
    echo Please build the Release configuration first (Option 2).
    echo %date% %time% - simple_test.exe not found >> "%log_file%"
    pause
    goto MAIN_MENU
)

echo %GREEN%Running main executable (cgrunner.exe)...%RESET%
echo %CYAN%=========================================%RESET%
"%build_dir%\bin\Release\cgrunner.exe"
echo.

echo %GREEN%Running example executable (simple_test.exe)...%RESET%
echo %CYAN%=========================================%RESET%
"%build_dir%\bin\Release\simple_test.exe"
echo.

echo %GREEN%SUCCESS: All Release executables completed!%RESET%
echo %date% %time% - Release executables completed successfully >> "%log_file%"
pause
goto MAIN_MENU

:BUILD_AND_RUN_DEBUG
cls
echo.
echo %CYAN%================================================%RESET%
echo %CYAN%     Building and Running Debug Configuration%RESET%
echo %CYAN%================================================%RESET%
echo.

echo %date% %time% - Starting Build and Run Debug >> "%log_file%"

REM Clean previous build
if exist %build_dir% (
    echo %YELLOW%Cleaning previous build...%RESET%
    rmdir /s /q %build_dir%
    echo %date% %time% - Cleaned previous build >> "%log_file%"
)

REM Configure project
echo %BLUE%Configuring project with !compiler_name!...%RESET%
"%compiler_path%" -B %build_dir% -G "Visual Studio 16 2019" 2>&1 | tee -a "%log_file%"
if %errorlevel% neq 0 (
    echo.
    echo %RED%ERROR: Configuration failed!%RESET%
    echo %date% %time% - Configuration failed >> "%log_file%"
    pause
    goto MAIN_MENU
)

REM Build Debug configuration
echo.
echo %BLUE%Building Debug configuration...%RESET%
"%compiler_path%" --build %build_dir% --config Debug 2>&1 | tee -a "%log_file%"
if %errorlevel% neq 0 (
    echo.
    echo %RED%ERROR: Build failed!%RESET%
    echo %date% %time% - Build failed >> "%log_file%"
    pause
    goto MAIN_MENU
)

echo.
echo %GREEN%SUCCESS: Debug build completed!%RESET%
echo.
echo %GREEN%Running executables...%RESET%
echo %CYAN%=========================================%RESET%

echo %GREEN%Running main executable (cgrunner.exe)...%RESET%
echo %CYAN%=========================================%RESET%
"%build_dir%\bin\Debug\cgrunner.exe"
echo.

echo %GREEN%Running example executable (simple_test.exe)...%RESET%
echo %CYAN%=========================================%RESET%
"%build_dir%\bin\Debug\simple_test.exe"
echo.

echo %GREEN%SUCCESS: Build and run completed!%RESET%
echo %date% %time% - Build and run debug completed successfully >> "%log_file%"
pause
goto MAIN_MENU

:BUILD_AND_RUN_RELEASE
cls
echo.
echo %CYAN%================================================%RESET%
echo %CYAN%    Building and Running Release Configuration%RESET%
echo %CYAN%================================================%RESET%
echo.

echo %date% %time% - Starting Build and Run Release >> "%log_file%"

REM Clean previous build
if exist %build_dir% (
    echo %YELLOW%Cleaning previous build...%RESET%
    rmdir /s /q %build_dir%
    echo %date% %time% - Cleaned previous build >> "%log_file%"
)

REM Configure project
echo %BLUE%Configuring project with !compiler_name!...%RESET%
"%compiler_path%" -B %build_dir% -G "Visual Studio 16 2019" 2>&1 | tee -a "%log_file%"
if %errorlevel% neq 0 (
    echo.
    echo %RED%ERROR: Configuration failed!%RESET%
    echo %date% %time% - Configuration failed >> "%log_file%"
    pause
    goto MAIN_MENU
)

REM Build Release configuration
echo.
echo %BLUE%Building Release configuration...%RESET%
"%compiler_path%" --build %build_dir% --config Release 2>&1 | tee -a "%log_file%"
if %errorlevel% neq 0 (
    echo.
    echo %RED%ERROR: Build failed!%RESET%
    echo %date% %time% - Build failed >> "%log_file%"
    pause
    goto MAIN_MENU
)

echo.
echo %GREEN%SUCCESS: Release build completed!%RESET%
echo.
echo %GREEN%Running executables...%RESET%
echo %CYAN%=========================================%RESET%

echo %GREEN%Running main executable (cgrunner.exe)...%RESET%
echo %CYAN%=========================================%RESET%
"%build_dir%\bin\Release\cgrunner.exe"
echo.

echo %GREEN%Running example executable (simple_test.exe)...%RESET%
echo %CYAN%=========================================%RESET%
"%build_dir%\bin\Release\simple_test.exe"
echo.

echo %GREEN%SUCCESS: Build and run completed!%RESET%
echo %date% %time% - Build and run release completed successfully >> "%log_file%"
pause
goto MAIN_MENU

:CLEAN_BUILD
cls
echo.
echo %CYAN%================================================%RESET%
echo %CYAN%           Cleaning Build Directory%RESET%
echo %CYAN%================================================%RESET%
echo.

echo %date% %time% - Cleaning build directory >> "%log_file%"

if exist %build_dir% (
    echo %YELLOW%Cleaning build directory...%RESET%
    rmdir /s /q %build_dir%
    echo %GREEN%SUCCESS: Build directory cleaned!%RESET%
    echo %date% %time% - Build directory cleaned successfully >> "%log_file%"
) else (
    echo %YELLOW%Build directory does not exist.%RESET%
    echo %date% %time% - Build directory did not exist >> "%log_file%"
)

echo.
pause
goto MAIN_MENU

:SHOW_STATUS
cls
echo.
echo %CYAN%================================================%RESET%
echo %CYAN%            Project Status%RESET%
echo %CYAN%================================================%RESET%
echo.

echo %WHITE%Project Directory: %CD%%RESET%
echo %WHITE%Compiler: %compiler_name%%RESET%
echo %WHITE%Build Directory: %build_dir%%RESET%
echo.

echo %BLUE%Build Directory Status:%RESET%
if exist %build_dir% (
    echo %GREEN%[EXISTS]%RESET% %build_dir%\
    if exist "%build_dir%\bin\Debug\cgrunner.exe" (
        echo %GREEN%[EXISTS]%RESET% %build_dir%\bin\Debug\cgrunner.exe
    ) else (
        echo %RED%[MISSING]%RESET% %build_dir%\bin\Debug\cgrunner.exe
    )
    if exist "%build_dir%\bin\Debug\simple_test.exe" (
        echo %GREEN%[EXISTS]%RESET% %build_dir%\bin\Debug\simple_test.exe
    ) else (
        echo %RED%[MISSING]%RESET% %build_dir%\bin\Debug\simple_test.exe
    )
    if exist "%build_dir%\bin\Release\cgrunner.exe" (
        echo %GREEN%[EXISTS]%RESET% %build_dir%\bin\Release\cgrunner.exe
    ) else (
        echo %RED%[MISSING]%RESET% %build_dir%\bin\Release\cgrunner.exe
    )
    if exist "%build_dir%\bin\Release\simple_test.exe" (
        echo %GREEN%[EXISTS]%RESET% %build_dir%\bin\Release\simple_test.exe
    ) else (
        echo %RED%[MISSING]%RESET% %build_dir%\bin\Release\simple_test.exe
    )
) else (
    echo %RED%[MISSING]%RESET% %build_dir%\
)

echo.
echo %BLUE%Source Files:%RESET%
if exist "src\main.cpp" (
    echo %GREEN%[EXISTS]%RESET% src\main.cpp
) else (
    echo %RED%[MISSING]%RESET% src\main.cpp
)
if exist "examples\simple_test.cpp" (
    echo %GREEN%[EXISTS]%RESET% examples\simple_test.cpp
) else (
    echo %RED%[MISSING]%RESET% examples\simple_test.cpp
)

echo.
echo %BLUE%CMake Configuration:%RESET%
if exist "CMakeLists.txt" (
    echo %GREEN%[EXISTS]%RESET% CMakeLists.txt
) else (
    echo %RED%[MISSING]%RESET% CMakeLists.txt
)

echo.
echo %BLUE%Test Files:%RESET%
if exist "tests\test_tensor.cpp" (
    echo %GREEN%[EXISTS]%RESET% tests\test_tensor.cpp
) else (
    echo %RED%[MISSING]%RESET% tests\test_tensor.cpp
)
if exist "tests\test_autograd.cpp" (
    echo %GREEN%[EXISTS]%RESET% tests\test_autograd.cpp
) else (
    echo %RED%[MISSING]%RESET% tests\test_autograd.cpp
)

echo.
pause
goto MAIN_MENU

:OPEN_VS_SOLUTION
cls
echo.
echo %CYAN%================================================%RESET%
echo %CYAN%        Opening Visual Studio Solution%RESET%
echo %CYAN%================================================%RESET%
echo.

if not exist "%build_dir%\CGROOT++.sln" (
    echo %RED%ERROR: Visual Studio solution not found!%RESET%
    echo Please build the project first (Options 1 or 2).
    echo %date% %time% - VS solution not found >> "%log_file%"
    pause
    goto MAIN_MENU
)

echo %BLUE%Opening Visual Studio solution...%RESET%
start "" "%build_dir%\CGROOT++.sln"
echo %GREEN%SUCCESS: Visual Studio solution opened!%RESET%
echo %date% %time% - VS solution opened >> "%log_file%"
pause
goto MAIN_MENU

:RUN_TESTS
cls
echo.
echo %CYAN%================================================%RESET%
echo %CYAN%              Running Tests%RESET%
echo %CYAN%================================================%RESET%
echo.

echo %date% %time% - Running tests >> "%log_file%"

REM Check if test executables exist
set "test_found=0"
if exist "%build_dir%\bin\Debug\cgrunner.exe" (
    set "test_found=1"
    echo %GREEN%Running main executable tests...%RESET%
    echo %CYAN%=========================================%RESET%
    "%build_dir%\bin\Debug\cgrunner.exe"
    echo.
)

if exist "%build_dir%\bin\Debug\simple_test.exe" (
    set "test_found=1"
    echo %GREEN%Running simple test executable...%RESET%
    echo %CYAN%=========================================%RESET%
    "%build_dir%\bin\Debug\simple_test.exe"
    echo.
)

if %test_found% EQU 0 (
    echo %RED%ERROR: No test executables found!%RESET%
    echo Please build the Debug configuration first Option 1.
    echo %date% %time% - No test executables found >> "%log_file%"
    pause
    goto MAIN_MENU
)

echo %GREEN%SUCCESS: All tests completed!%RESET%
echo %date% %time% - Tests completed successfully >> "%log_file%"
pause
goto MAIN_MENU

:GENERATE_DOCS
cls
echo.
echo %CYAN%================================================%RESET%
echo %CYAN%         Generating Documentation%RESET%
echo %CYAN%================================================%RESET%
echo.

echo %date% %time% - Generating documentation >> "%log_file%"

echo %BLUE%Creating documentation directory...%RESET%
if not exist "docs" mkdir docs

echo %BLUE%Generating source code documentation...%RESET%
echo # CGROOT++ Documentation > docs\README.md
echo. >> docs\README.md
echo Generated on %date% %time% >> docs\README.md
echo. >> docs\README.md
echo ## Project Structure >> docs\README.md
echo. >> docs\README.md
echo ### Core Components >> docs\README.md
echo - **tensor.h/cpp**: Core tensor operations >> docs\README.md
echo - **parameter.h**: Parameter management >> docs\README.md
echo - **shape.h**: Shape handling >> docs\README.md
echo. >> docs\README.md
echo ### Neural Network Components >> docs\README.md
echo - **linear.h**: Linear layer implementation >> docs\README.md
echo - **relu.h**: ReLU activation function >> docs\README.md
echo - **sigmoid.h**: Sigmoid activation function >> docs\README.md
echo - **conv2d.h**: 2D Convolution layer >> docs\README.md
echo - **sequential.h**: Sequential model container >> docs\README.md
echo. >> docs\README.md
echo ### Loss Functions >> docs\README.md
echo - **mse_loss.h**: Mean Squared Error loss >> docs\README.md
echo - **cross_entropy_loss.h**: Cross Entropy loss >> docs\README.md
echo. >> docs\README.md
echo ### Optimizers >> docs\README.md
echo - **sgd.h**: Stochastic Gradient Descent >> docs\README.md
echo - **adam.h**: Adam optimizer >> docs\README.md
echo. >> docs\README.md
echo ### Math Operations >> docs\README.md
echo - **cpu_kernels.h/cpp**: CPU-based mathematical operations >> docs\README.md
echo. >> docs\README.md
echo ### Autograd System >> docs\README.md
echo - **graph.h/cpp**: Computational graph >> docs\README.md
echo - **op_nodes.h/cpp**: Operation nodes >> docs\README.md

echo %GREEN%SUCCESS: Documentation generated in docs\ directory!%RESET%
echo %date% %time% - Documentation generated successfully >> "%log_file%"
pause
goto MAIN_MENU

:SHOW_LOG
cls
echo.
echo %CYAN%================================================%RESET%
echo %CYAN%              Build Log%RESET%
echo %CYAN%================================================%RESET%
echo.

if exist "%log_file%" (
    echo %BLUE%Contents of %log_file%:%RESET%
    echo %CYAN%=========================================%RESET%
    type "%log_file%"
    echo %CYAN%=========================================%RESET%
) else (
    echo %YELLOW%No build log found.%RESET%
)

echo.
pause
goto MAIN_MENU

:EXIT
cls
echo.
echo %GREEN%Thank you for using %project_name% Project Manager!%RESET%
echo %date% %time% - Manager closed >> "%log_file%"
echo.
exit /b 0