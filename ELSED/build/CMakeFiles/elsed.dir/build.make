# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hyunjun/line_ws/src/UV-SLAM/ELSED

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hyunjun/line_ws/src/UV-SLAM/ELSED/build

# Include any dependencies generated for this target.
include CMakeFiles/elsed.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/elsed.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/elsed.dir/flags.make

CMakeFiles/elsed.dir/src/ELSED.cpp.o: CMakeFiles/elsed.dir/flags.make
CMakeFiles/elsed.dir/src/ELSED.cpp.o: ../src/ELSED.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hyunjun/line_ws/src/UV-SLAM/ELSED/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/elsed.dir/src/ELSED.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/elsed.dir/src/ELSED.cpp.o -c /home/hyunjun/line_ws/src/UV-SLAM/ELSED/src/ELSED.cpp

CMakeFiles/elsed.dir/src/ELSED.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/elsed.dir/src/ELSED.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hyunjun/line_ws/src/UV-SLAM/ELSED/src/ELSED.cpp > CMakeFiles/elsed.dir/src/ELSED.cpp.i

CMakeFiles/elsed.dir/src/ELSED.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/elsed.dir/src/ELSED.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hyunjun/line_ws/src/UV-SLAM/ELSED/src/ELSED.cpp -o CMakeFiles/elsed.dir/src/ELSED.cpp.s

CMakeFiles/elsed.dir/src/ELSED.cpp.o.requires:

.PHONY : CMakeFiles/elsed.dir/src/ELSED.cpp.o.requires

CMakeFiles/elsed.dir/src/ELSED.cpp.o.provides: CMakeFiles/elsed.dir/src/ELSED.cpp.o.requires
	$(MAKE) -f CMakeFiles/elsed.dir/build.make CMakeFiles/elsed.dir/src/ELSED.cpp.o.provides.build
.PHONY : CMakeFiles/elsed.dir/src/ELSED.cpp.o.provides

CMakeFiles/elsed.dir/src/ELSED.cpp.o.provides.build: CMakeFiles/elsed.dir/src/ELSED.cpp.o


CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.o: CMakeFiles/elsed.dir/flags.make
CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.o: ../src/EdgeDrawer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hyunjun/line_ws/src/UV-SLAM/ELSED/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.o -c /home/hyunjun/line_ws/src/UV-SLAM/ELSED/src/EdgeDrawer.cpp

CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hyunjun/line_ws/src/UV-SLAM/ELSED/src/EdgeDrawer.cpp > CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.i

CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hyunjun/line_ws/src/UV-SLAM/ELSED/src/EdgeDrawer.cpp -o CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.s

CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.o.requires:

.PHONY : CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.o.requires

CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.o.provides: CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.o.requires
	$(MAKE) -f CMakeFiles/elsed.dir/build.make CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.o.provides.build
.PHONY : CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.o.provides

CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.o.provides.build: CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.o


CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.o: CMakeFiles/elsed.dir/flags.make
CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.o: ../src/FullSegmentInfo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hyunjun/line_ws/src/UV-SLAM/ELSED/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.o -c /home/hyunjun/line_ws/src/UV-SLAM/ELSED/src/FullSegmentInfo.cpp

CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hyunjun/line_ws/src/UV-SLAM/ELSED/src/FullSegmentInfo.cpp > CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.i

CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hyunjun/line_ws/src/UV-SLAM/ELSED/src/FullSegmentInfo.cpp -o CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.s

CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.o.requires:

.PHONY : CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.o.requires

CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.o.provides: CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.o.requires
	$(MAKE) -f CMakeFiles/elsed.dir/build.make CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.o.provides.build
.PHONY : CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.o.provides

CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.o.provides.build: CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.o


CMakeFiles/elsed.dir/src/main.cpp.o: CMakeFiles/elsed.dir/flags.make
CMakeFiles/elsed.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hyunjun/line_ws/src/UV-SLAM/ELSED/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/elsed.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/elsed.dir/src/main.cpp.o -c /home/hyunjun/line_ws/src/UV-SLAM/ELSED/src/main.cpp

CMakeFiles/elsed.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/elsed.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hyunjun/line_ws/src/UV-SLAM/ELSED/src/main.cpp > CMakeFiles/elsed.dir/src/main.cpp.i

CMakeFiles/elsed.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/elsed.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hyunjun/line_ws/src/UV-SLAM/ELSED/src/main.cpp -o CMakeFiles/elsed.dir/src/main.cpp.s

CMakeFiles/elsed.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/elsed.dir/src/main.cpp.o.requires

CMakeFiles/elsed.dir/src/main.cpp.o.provides: CMakeFiles/elsed.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/elsed.dir/build.make CMakeFiles/elsed.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/elsed.dir/src/main.cpp.o.provides

CMakeFiles/elsed.dir/src/main.cpp.o.provides.build: CMakeFiles/elsed.dir/src/main.cpp.o


# Object files for target elsed
elsed_OBJECTS = \
"CMakeFiles/elsed.dir/src/ELSED.cpp.o" \
"CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.o" \
"CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.o" \
"CMakeFiles/elsed.dir/src/main.cpp.o"

# External object files for target elsed
elsed_EXTERNAL_OBJECTS =

libelsed.a: CMakeFiles/elsed.dir/src/ELSED.cpp.o
libelsed.a: CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.o
libelsed.a: CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.o
libelsed.a: CMakeFiles/elsed.dir/src/main.cpp.o
libelsed.a: CMakeFiles/elsed.dir/build.make
libelsed.a: CMakeFiles/elsed.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hyunjun/line_ws/src/UV-SLAM/ELSED/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX static library libelsed.a"
	$(CMAKE_COMMAND) -P CMakeFiles/elsed.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/elsed.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/elsed.dir/build: libelsed.a

.PHONY : CMakeFiles/elsed.dir/build

CMakeFiles/elsed.dir/requires: CMakeFiles/elsed.dir/src/ELSED.cpp.o.requires
CMakeFiles/elsed.dir/requires: CMakeFiles/elsed.dir/src/EdgeDrawer.cpp.o.requires
CMakeFiles/elsed.dir/requires: CMakeFiles/elsed.dir/src/FullSegmentInfo.cpp.o.requires
CMakeFiles/elsed.dir/requires: CMakeFiles/elsed.dir/src/main.cpp.o.requires

.PHONY : CMakeFiles/elsed.dir/requires

CMakeFiles/elsed.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/elsed.dir/cmake_clean.cmake
.PHONY : CMakeFiles/elsed.dir/clean

CMakeFiles/elsed.dir/depend:
	cd /home/hyunjun/line_ws/src/UV-SLAM/ELSED/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hyunjun/line_ws/src/UV-SLAM/ELSED /home/hyunjun/line_ws/src/UV-SLAM/ELSED /home/hyunjun/line_ws/src/UV-SLAM/ELSED/build /home/hyunjun/line_ws/src/UV-SLAM/ELSED/build /home/hyunjun/line_ws/src/UV-SLAM/ELSED/build/CMakeFiles/elsed.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/elsed.dir/depend

