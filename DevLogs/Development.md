### June 26

- Tested Galaxy Engine from https://github.com/NarcisCalin/Galaxy-Engine (C++, only avaliable for Windows) and built the executable exe file
Notes:
1. For building it in VS Code, need to download and install MinGW-w64 from WinLibs (https://www.winlibs.com/)
2. Install the plugins that VS Code recommanded (for C++)
3. Add a line to the environment variables (D:\mingw64\bin)
4. `yaml.cmake`: `GIT_TAG 0.8.0` -> `GIT_TAG master`
5. Run the following code in terminal (one by one)
 ```bash
   mkdir build
   cd build
   cmake .. -G "MinGW Makefiles" -DBUILD_AUDIO=OFF -DSUPPORT_AUDIO=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
   cmake --build .
```
6. Ensure the `/Textures`,`/Shaders`,`/fonts`,`/Sounds`folders in `/build` (I was trapped by missing `/Sounds`)
7. Then the `GalaxyEngine.exe` in `/build` should be executable
   
- Started considering how to combine it with Python code
