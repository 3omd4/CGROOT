# GUI sources (only build if Qt6 is available)

set(GUI_WIDGETS_SOURCES
    widgets/configurationwidget.cpp
    widgets/configurationwidget.h
    widgets/imageviewerwidget.cpp
    widgets/imageviewerwidget.h
    widgets/inferencewidget.cpp
    widgets/inferencewidget.h
    widgets/metricswidget.cpp
    widgets/metricswidget.h
    widgets/trainingwidget.cpp
    widgets/trainingwidget.h
)

set(GUI_CONTROLLERS_SOURCES
    controllers/modelcontroller.cpp
    controllers/modelcontroller.h
)

set(GUI_MAIN_SOURCES
    mainwindow.cpp
    mainwindow.h
    main.cpp
)

set(GUI_SOURCES
    ${GUI_WIDGETS_SOURCES}
    ${GUI_CONTROLLERS_SOURCES}
    ${GUI_MAIN_SOURCES}
)

# Filter GUI sources to only include existing files
set(FILTERED_GUI_SOURCES "")
foreach(GUI_FILE ${GUI_SOURCES})
    set(FULL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${GUI_FILE})
    if(${GUI_FILE} MATCHES "\\.h$")
        list(APPEND FILTERED_GUI_SOURCES ${FULL_PATH})
    elseif(EXISTS ${FULL_PATH})
        list(APPEND FILTERED_GUI_SOURCES ${FULL_PATH})
    endif()
endforeach()

# Create GUI executable
add_executable(cgroot_gui ${FILTERED_GUI_SOURCES})
target_link_libraries(cgroot_gui 
    cgroot_lib
    Qt6::Core
    Qt6::Widgets
    Qt6::Charts
)

set_target_properties(cgroot_gui PROPERTIES
    WIN32_EXECUTABLE TRUE
)

