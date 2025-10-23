@echo off
setlocal enabledelayedexpansion
title CGROOT++ Project Manager

:MAIN_MENU
cls
echo.
echo ================================================
echo           CGROOT++ Project Manager
echo ================================================
echo.
echo Choose an option:
echo.
echo 1. Build Debug Configuration
echo 2. Build Release Configuration
echo 3. Run Debug Executables
echo 4. Run Release Executables
echo 5. Build and Run Debug
echo 6. Build and Run Release
echo 7. Clean Build Directory
echo 8. Show Project Status
echo 9. Open Visual Studio Solution
echo 0. Exit
echo.
set /p choice="Enter your choice (0-9): "

if "%choice%"=="1" goto BUILD_DEBUG
if "%choice%"=="2" goto BUILD_RELEASE
if "%choice%"=="3" goto RUN_DEBUG
if "%choice%"=="4" goto RUN_RELEASE
if "%choice%"=="5" goto BUILD_AND_RUN_DEBUG
if "%choice%"=="6" goto BUILD_AND_RUN_RELEASE
if "%choice%"=="7" goto CLEAN_BUILD
if "%choice%"=="8" goto SHOW_STATUS
if "%choice%"=="9" goto OPEN_VS_SOLUTION
if "%choice%"=="0" goto EXIT
echo Invalid choice! Please try again.
pause
goto MAIN_MENU

:BUILD_DEBUG
cls
echo ================================================
echo        Building Debug Configuration
echo ================================================
echo.

REM Clean previous build
if exist build (
    echo Cleaning previous build...
    rmdir /s /q build
)

REM Configure with Visual Studio 2019
echo Configuring project with Visual Studio 2019...
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -B build -G "Visual Studio 16 2019"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Configuration failed!
    echo Please check your Visual Studio installation.
    pause
    goto MAIN_MENU
)

REM Build Debug configuration
echo.
echo Building Debug configuration...
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Debug
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Build failed!
    pause
    goto MAIN_MENU
)

echo.
echo SUCCESS: Debug build completed!
echo.
echo Executables created:
echo - build\bin\Debug\cgrunner.exe
echo - build\bin\Debug\simple_test.exe
echo.
pause
goto MAIN_MENU

:BUILD_RELEASE
cls
echo ================================================
echo       Building Release Configuration
echo ================================================
echo.

REM Clean previous build
if exist build (
    echo Cleaning previous build...
    rmdir /s /q build
)

REM Configure with Visual Studio 2019
echo Configuring project with Visual Studio 2019...
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -B build -G "Visual Studio 16 2019"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Configuration failed!
    echo Please check your Visual Studio installation.
    pause
    goto MAIN_MENU
)

REM Build Release configuration
echo.
echo Building Release configuration...
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Release
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Build failed!
    pause
    goto MAIN_MENU
)

echo.
echo SUCCESS: Release build completed!
echo.
echo Executables created:
echo - build\bin\Release\cgrunner.exe
echo - build\bin\Release\simple_test.exe
echo.
pause
goto MAIN_MENU

:RUN_DEBUG
cls
echo ================================================
echo        Running Debug Executables
echo ================================================
echo.

REM Check if executables exist
if not exist "build\bin\Debug\cgrunner.exe" (
    echo ERROR: cgrunner.exe not found!
    echo Please build the Debug configuration first (Option 1).
    pause
    goto MAIN_MENU
)

if not exist "build\bin\Debug\simple_test.exe" (
    echo ERROR: simple_test.exe not found!
    echo Please build the Debug configuration first (Option 1).
    pause
    goto MAIN_MENU
)

echo Running main executable (cgrunner.exe)...
echo ==========================================
build\bin\Debug\cgrunner.exe
echo.

echo Running example executable (simple_test.exe)...
echo ==========================================
build\bin\Debug\simple_test.exe
echo.

echo SUCCESS: All Debug executables completed!
pause
goto MAIN_MENU

:RUN_RELEASE
cls
echo ================================================
echo       Running Release Executables
echo ================================================
echo.

REM Check if executables exist
if not exist "build\bin\Release\cgrunner.exe" (
    echo ERROR: cgrunner.exe not found!
    echo Please build the Release configuration first (Option 2).
    pause
    goto MAIN_MENU
)

if not exist "build\bin\Release\simple_test.exe" (
    echo ERROR: simple_test.exe not found!
    echo Please build the Release configuration first (Option 2).
    pause
    goto MAIN_MENU
)

echo Running main executable (cgrunner.exe)...
echo ==========================================
build\bin\Release\cgrunner.exe
echo.

echo Running example executable (simple_test.exe)...
echo ==========================================
build\bin\Release\simple_test.exe
echo.

echo SUCCESS: All Release executables completed!
pause
goto MAIN_MENU

:BUILD_AND_RUN_DEBUG
cls
echo ================================================
echo     Building and Running Debug Configuration
echo ================================================
echo.

REM Clean previous build
if exist build (
    echo Cleaning previous build...
    rmdir /s /q build
)

REM Configure with Visual Studio 2019
echo Configuring project with Visual Studio 2019...
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -B build -G "Visual Studio 16 2019"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Configuration failed!
    pause
    goto MAIN_MENU
)

REM Build Debug configuration
echo.
echo Building Debug configuration...
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Debug
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Build failed!
    pause
    goto MAIN_MENU
)

echo.
echo SUCCESS: Debug build completed!
echo.
echo Running executables...
echo ==========================================

echo Running main executable (cgrunner.exe)...
echo ==========================================
build\bin\Debug\cgrunner.exe
echo.

echo Running example executable (simple_test.exe)...
echo ==========================================
build\bin\Debug\simple_test.exe
echo.

echo SUCCESS: Build and run completed!
pause
goto MAIN_MENU

:BUILD_AND_RUN_RELEASE
cls
echo ================================================
echo    Building and Running Release Configuration
echo ================================================
echo.

REM Clean previous build
if exist build (
    echo Cleaning previous build...
    rmdir /s /q build
)

REM Configure with Visual Studio 2019
echo Configuring project with Visual Studio 2019...
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -B build -G "Visual Studio 16 2019"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Configuration failed!
    pause
    goto MAIN_MENU
)

REM Build Release configuration
echo.
echo Building Release configuration...
"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Release
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Build failed!
    pause
    goto MAIN_MENU
)

echo.
echo SUCCESS: Release build completed!
echo.
echo Running executables...
echo ==========================================

echo Running main executable (cgrunner.exe)...
echo ==========================================
build\bin\Release\cgrunner.exe
echo.

echo Running example executable (simple_test.exe)...
echo ==========================================
build\bin\Release\simple_test.exe
echo.

echo SUCCESS: Build and run completed!
pause
goto MAIN_MENU

:CLEAN_BUILD
cls
echo ================================================
echo           Cleaning Build Directory
echo ================================================
echo.

if exist build (
    echo Cleaning build directory...
    rmdir /s /q build
    echo SUCCESS: Build directory cleaned!
) else (
    echo Build directory does not exist.
)

echo.
pause
goto MAIN_MENU

:SHOW_STATUS
cls
echo ================================================
echo            Project Status
echo ================================================
echo.

echo Project Directory: %CD%
echo.

echo Build Directory Status:
if exist build (
    echo [EXISTS] build\
    if exist "build\bin\Debug\cgrunner.exe" (
        echo [EXISTS] build\bin\Debug\cgrunner.exe
    ) else (
        echo [MISSING] build\bin\Debug\cgrunner.exe
    )
    if exist "build\bin\Debug\simple_test.exe" (
        echo [EXISTS] build\bin\Debug\simple_test.exe
    ) else (
        echo [MISSING] build\bin\Debug\simple_test.exe
    )
    if exist "build\bin\Release\cgrunner.exe" (
        echo [EXISTS] build\bin\Release\cgrunner.exe
    ) else (
        echo [MISSING] build\bin\Release\cgrunner.exe
    )
    if exist "build\bin\Release\simple_test.exe" (
        echo [EXISTS] build\bin\Release\simple_test.exe
    ) else (
        echo [MISSING] build\bin\Release\simple_test.exe
    )
) else (
    echo [MISSING] build\
)

echo.
echo Source Files:
if exist "src\main.cpp" (
    echo [EXISTS] src\main.cpp
) else (
    echo [MISSING] src\main.cpp
)
if exist "examples\simple_test.cpp" (
    echo [EXISTS] examples\simple_test.cpp
) else (
    echo [MISSING] examples\simple_test.cpp
)

echo.
echo CMake Configuration:
if exist "CMakeLists.txt" (
    echo [EXISTS] CMakeLists.txt
) else (
    echo [MISSING] CMakeLists.txt
)

echo.
pause
goto MAIN_MENU

:OPEN_VS_SOLUTION
cls
echo ================================================
echo        Opening Visual Studio Solution
echo ================================================
echo.

if not exist "build\CGROOT++.sln" (
    echo ERROR: Visual Studio solution not found!
    echo Please build the project first (Options 1 or 2).
    pause
    goto MAIN_MENU
)

echo Opening Visual Studio solution...
start "" "build\CGROOT++.sln"
echo SUCCESS: Visual Studio solution opened!
pause
goto MAIN_MENU

:EXIT
cls
echo.
echo Thank you for using CGROOT++ Project Manager!
echo.
exit /b 0
