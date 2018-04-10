
if(system_cmake_included)
  return()
endif()
set(system_cmake_included true)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# TODO: vnni
if(WITH_VNNI)
  set(CMAKE_CCXX_FLAGS "${CMAKE_CCXX_FLAGS} -mavx512f -mavx512cd -mavx512vl -mavx512bw -mavx512dq -mavx512vnni")
endif()

# external dependencies log output
set(external_project_dependencies)
set(EXTERNAL_PROJECT_LOG_ARGS
    LOG_DOWNLOAD    0     # Wrap download in script to log output
    LOG_UPDATE      1     # Wrap update in script to log output
    LOG_CONFIGURE   1     # Wrap configure in script to log output
    LOG_BUILD       0     # Wrap build in script to log output
    LOG_TEST        1     # Wrap test in script to log output
    LOG_INSTALL     0     # Wrap install in script to log output
)
