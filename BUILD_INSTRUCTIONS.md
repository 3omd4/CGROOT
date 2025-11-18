# Build Instructions

## Quick Build (Recommended)

### Option 1: Using Visual Studio (Easiest)
1. Open `CGroot++.sln` in Visual Studio 2019 or later
2. Select configuration: **Debug** or **Release** (top toolbar)
3. Right-click on solution → **Build Solution** (or press `Ctrl+Shift+B`)
4. Executables will be in:
   - `bin/Debug/` for Debug builds
   - `bin/Release/` for Release builds

### Option 2: Using CMake Command Line
```powershell
# Build Debug configuration
cmake --build . --config Debug

# Build Release configuration  
cmake --build . --config Release
```

### Option 3: Using the Build Script
```powershell
# Run the project manager
.\scripts\CGROOT_Manager.bat
```
Then select option to build Debug or Release.

## Expected Executables

After building, you should find:
- `bin/Debug/cgrunner.exe` (or `bin/Release/cgrunner.exe`)
- `bin/Debug/simple_test.exe` (or `bin/Release/simple_test.exe`)
- `bin/Debug/xor_solver.exe` (or `bin/Release/xor_solver.exe`)
- `bin/Debug/cgroot_gui.exe` (if Qt6 is found)

## Troubleshooting

**If `bin/` directory doesn't exist after build:**
- Check for build errors in Visual Studio's Output window
- Verify all source files compile without errors
- Make sure you selected a valid configuration (Debug/Release)

**If executables are in wrong location:**
- You're building in-source (not recommended)
- For cleaner builds, use: `cmake -B build -S .` then `cmake --build build`

