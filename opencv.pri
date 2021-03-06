DEFINES += OPENCV_LIB

windows {
    TEST_OPENCV_INCLUDE_PATH = $$(OPENCV_INCLUDE_PATH)
    isEmpty(TEST_OPENCV_INCLUDE_PATH) {
        error("Variable \"OPENCV_INCLUDE_PATH\" is not set")
    } else {
        TEST_OPENCV_LIB_PATH = $$(OPENCV_LIB_PATH)
        isEmpty(TEST_OPENCV_LIB_PATH) {
            error("Variable \"OPENCV_LIB_PATH\" is not set")
        } else {
            INCLUDEPATH += $$(OPENCV_INCLUDE_PATH)
            DEPENDPATH += $$(OPENCV_INCLUDE_PATH)

            CONFIG(debug, release|debug) {
                OPENCV_VERSION=310d
            } else {
                OPENCV_VERSION=310
            }

            LIBS += -L$$(OPENCV_LIB_PATH) \
                -lopencv_core$$OPENCV_VERSION \
                -lopencv_imgproc$$OPENCV_VERSION \
                -lopencv_videoio$$OPENCV_VERSION \
                -lopencv_highgui$$OPENCV_VERSION \
                -lopencv_aruco$$OPENCV_VERSION
        }
    }
} else: unix {
    TEST_OPENCV_INCLUDE_PATH = $$(OPENCV_INCLUDE_PATH)
    isEmpty(TEST_OPENCV_INCLUDE_PATH) {
        error("Variable \"OPENCV_INCLUDE_PATH\" is not set")
    } else {
        TEST_OPENCV_LIB_PATH = $$(OPENCV_LIB_PATH)
        isEmpty(TEST_OPENCV_LIB_PATH) {
            error("Variable \"OPENCV_LIB_PATH\" is not set")
        } else {
            INCLUDEPATH += $$(OPENCV_INCLUDE_PATH)
            DEPENDPATH += $$(OPENCV_INCLUDE_PATH)

            CONFIG(debug, release|debug) {
                OPENCV_VERSION=310d
            } else {
                OPENCV_VERSION=310
            }

            LIBS += -L$$(OPENCV_LIB_PATH) \
                -lopencv_core \
                -lopencv_imgproc \
                -lopencv_videoio \
                -lopencv_highgui \
                -lopencv_aruco \
                -lopencv_imgcodecs
        }
    }
} else {
    error("OpenCV not included")
}
