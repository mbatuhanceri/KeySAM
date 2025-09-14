cmake_minimum_required(VERSION 3.30)
project(libtorch_segmentation)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_TOOLCHAIN_FILE "C:/Users/PC/.vcpkg-clion/vcpkg_peoplenet/scripts/buildsystems/vcpkg.cmake")

set(CMAKE_PREFIX_PATH
        "C:/tools/libtorch/share/cmake"
        "C:/tools/opencv-cpp/opencv/build"
        "C:/tools/torchvision-cpp"
)

set(PNG_PNG_INCLUDE_DIR "C:/Users/PC/.vcpkg-clion/vcpkg_peoplenet/installed/x64-windows/include")
set(PNG_LIBRARY "C:/Users/PC/.vcpkg-clion/vcpkg_peoplenet/installed/x64-windows/lib/libpng16.lib")


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(TorchVision_DIR "C:/tools/torchvision-cpp/share/cmake/TorchVision")
find_package(TorchVision REQUIRED)

message(STATUS "TorchVision found: ${TorchVision_FOUND}")
message(STATUS "TorchVision include dirs: ${TorchVision_INCLUDE_DIRS}")
message(STATUS "TorchVision libraries: ${TorchVision_LIBRARIES}")

find_package(OpenCV REQUIRED)

add_executable(libtorch_segmentation
        src/main.cpp
        src/utils/ConfigManager.cpp
        src/utils/ConfigManager.h
        src/models/IModel.cpp
        src/models/IModel.h
        src/models/PoseModel.cpp
        src/models/PoseModel.h
        src/modelProcessors/PoseModelProcessor.cpp
        src/modelProcessors/PoseModelProcessor.h
        src/pipelines/HumanSegmenterPipeline.cpp
        src/pipelines/HumanSegmenterPipeline.h
        src/pipelines/IPipeline.cpp
        src/pipelines/IPipeline.h
        src/utils/VideoUtils.cpp
        src/utils/VideoUtils.h
        src/utils/CVUtils.cpp
        src/utils/CVUtils.h
        src/utils/TensorUtils.cpp
        src/utils/TensorUtils.h
)

target_include_directories(libtorch_segmentation PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include
        ${JPEG_INCLUDE_DIR}
        "C:/tools/torchvision-cpp/include"
)

target_link_libraries(libtorch_segmentation
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        PNG::PNG
        ${JPEG_LIBRARIES}
        "C:/tools/torchvision-cpp/lib/torchvision.lib"
)

set_property(TARGET libtorch_segmentation PROPERTY CXX_STANDARD 17)

if (WIN32)
    file(GLOB TORCH_DLLS "C:/tools/libtorch/lib/*.dll")
    add_custom_command(TARGET libtorch_segmentation
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:libtorch_segmentation>)

    file(GLOB OPENCV_DLLS "C:/tools/opencv-cpp/opencv/build/x64/vc*/bin/*.dll")
    if(OPENCV_DLLS)
        add_custom_command(TARGET libtorch_segmentation
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${OPENCV_DLLS}
                $<TARGET_FILE_DIR:libtorch_segmentation>)
    endif()

    file(GLOB TORCHVISION_DLLS "C:/tools/torchvision-cpp/bin/*.dll")
    if(TORCHVISION_DLLS)
        add_custom_command(TARGET libtorch_segmentation
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCHVISION_DLLS}
                $<TARGET_FILE_DIR:libtorch_segmentation>)
    endif()
endif()