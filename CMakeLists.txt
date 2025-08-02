cmake_minimum_required(VERSION 3.30)
project(libtorch_segmentation)

set(CMAKE_CXX_STANDARD 17)

# Hem LibTorch hem OpenCV için PREFIX_PATH ayarla
set(CMAKE_PREFIX_PATH
        "C:/tools/libtorch/share/cmake"
        "C:/tools/opencv-cpp/opencv/build"
)

# LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# OpenCV
find_package(OpenCV REQUIRED)

# Debug: Dosya varlığını kontrol et
if(EXISTS "C:/tools/opencv-cpp/opencv/build/OpenCVConfig.cmake")
    message(STATUS "OpenCVConfig.cmake found!")
else()
    message(WARNING "OpenCVConfig.cmake not found at expected location")
    # Alternatif konumları kontrol et
    if(EXISTS "C:/tools/opencv-cpp/opencv/build/x64/vc16/lib/OpenCVConfig.cmake")
        message(STATUS "Found at x64/vc16/lib/")
        set(OpenCV_DIR "C:/tools/opencv-cpp/opencv/build/x64/vc16/lib")
    elseif(EXISTS "C:/tools/opencv-cpp/opencv/build/x64/vc15/lib/OpenCVConfig.cmake")
        message(STATUS "Found at x64/vc15/lib/")
        set(OpenCV_DIR "C:/tools/opencv-cpp/opencv/build/x64/vc15/lib")
    endif()
endif()

# Debug bilgileri
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

add_executable(libtorch_segmentation
        src/main.cpp
        src/utils/ConfigManager.cpp
        src/utils/ConfigManager.h
)

# Kütüphaneleri bağla
target_link_libraries(libtorch_segmentation ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(libtorch_segmentation PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/include
)
set_property(TARGET libtorch_segmentation PROPERTY CXX_STANDARD 17)

# Windows DLL kopyalama
if (WIN32)
    # LibTorch DLL'leri
    file(GLOB TORCH_DLLS "C:/tools/libtorch/lib/*.dll")
    add_custom_command(TARGET libtorch_segmentation
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:libtorch_segmentation>)

    # OpenCV DLL'leri
    file(GLOB OPENCV_DLLS "C:/tools/opencv-cpp/opencv/build/x64/vc*/bin/*.dll")
    if(OPENCV_DLLS)
        add_custom_command(TARGET libtorch_segmentation
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${OPENCV_DLLS}
                $<TARGET_FILE_DIR:libtorch_segmentation>)
    endif()
endif()