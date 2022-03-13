cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

project(astro)

find_package(VTK REQUIRED)

if (NOT VTK_FOUND)
  message("Skipping CylinderExample: ${VTK_NOT_FOUND_MESSAGE}")
  return()
endif()
message (STATUS "VTK_VERSION: ${VTK_VERSION}")
if (VTK_VERSION VERSION_LESS "8.90.0")
  # old system
  message("Not sure if this CMakeLists.txt file will work with a lesser VTK version.")
  return()
else()
  # Prevent a "command line is too long" failure in Windows.
  set(CMAKE_NINJA_FORCE_RESPONSE_FILE "ON" CACHE BOOL "Force Ninja to use response files.")
  add_executable(astro astro.cxx )
  target_link_libraries(astro PRIVATE ${VTK_LIBRARIES})
  # vtk_module_autoinit is needed
  vtk_module_autoinit(
    TARGETS astro
    MODULES ${VTK_LIBRARIES}
    )
endif()