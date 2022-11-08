import React, { useState, useEffect } from "react";
import { captureScreen } from "react-native-view-shot";
import {
  View,
  Dimensions,
  TouchableOpacity,
  StyleSheet,
  Image,
} from "react-native";
import Color from "./src/components/color";
import Stroke from "./src/components/stroke";
import useDrawingStore from "./src/store";
import constants from "./src/drawing/constants";
import utils from "./src/drawing/utils";
import ML from "./src/api/ml";
import {
  Gesture,
  GestureDetector,
  GestureHandlerRootView,
} from "react-native-gesture-handler";
import { Canvas, Path, Circle } from "@shopify/react-native-skia";
import { useSharedValue } from "react-native-reanimated";
import { Ionicons, MaterialCommunityIcons } from "@expo/vector-icons";
import Separator from "./src/components/Separator";

export default function Canv({ getter, setter, analyze }) {
  const [showBackground, setShowBackground] = useState(true);
  const paletteColors = ["black", "red", "blue", "green"];
  const [strokeWid, setStrokeWid] = useState(10);
  const [showStrokes, setShowStrokes] = useState(false);
  const setColor = useDrawingStore((state) => state.setColor);
  const [activePaletteColorIndex, setActivePaletteColorIndex] = useState(0);
  const [paths, setPaths] = useState([]);
  const [circles, setCircles] = useState([]);
  const [log, setLog] = useState([]);

  const [startedStroke, setStartedStroke] = useState(false);
  const [finishedStroke, setFinishedStroke] = useState(false);

  const paletteVisible = useSharedValue(false);

  const undoPath = () => {
    if (log[log.length - 1] == "c") {
      setCircles(circles.slice(0, -1));
    } else {
      setPaths(paths.slice(0, -1));
    }
    setLog(log.slice(0, -1));
  };

  const clearCanvas = () => {
    setPaths([]);
    setCircles([]);
    setLog([]);
  };

  // When Reanimated is installed, Gesture Handler will try to run on the UI thread
  // We can't do that here because we're accessing the component state, so we need set runOnJS(true)
  const pan = Gesture.Pan()
    .runOnJS(true)
    .onStart((g) => {
      const newPaths = [...paths];
      newPaths[paths.length] = {
        segments: [],
        color: paletteColors[activePaletteColorIndex],
      };
      newPaths[paths.length].segments.push(`M ${g.x} ${g.y}`);
      setPaths(newPaths);
      setLog([...log, "p"]);
      setStartedStroke(true);
      setFinishedStroke(false);
    })
    .onUpdate((g) => {
      const index = paths.length - 1;
      const newPaths = [...paths];
      if (newPaths?.[index]?.segments) {
        newPaths[index].segments.push(`L ${g.x} ${g.y}`);
        setPaths(newPaths);
      }
    })
    .onTouchesUp((g) => {
      const newPaths = [...paths];
      setPaths(newPaths);

      setStartedStroke(false);
      setFinishedStroke(true);
    })
    .minDistance(0);

  const tap = Gesture.Tap()
    .runOnJS(true)
    .onStart((g) => {
      setCircles([
        ...circles,
        {
          x: g.x,
          y: g.y,
          color: paletteColors[activePaletteColorIndex],
        },
      ]);
      setLog([...log, "c"]);
    });

  const MINUTE_MS = 1000;
  const sendScreenshot = () => {
    setShowBackground(false);
    captureScreen({
      result: "base64",
    }).then(
      async (string) => {
        try {
          setShowBackground(true);
          clearCanvas();
          const response = await ML.post("/upload-img", { base64: string });
          setter(getter + response.data);
        } catch (err) {
          console.error(err);
        }
      },
      (error) => console.error("Oops, Something Went Wrong", error)
    );
  };

  useEffect(() => {
    if (analyze) {
      const timer = setTimeout(sendScreenshot, MINUTE_MS);
      if (startedStroke == true) {
        clearTimeout(timer);
      }
      return () => clearTimeout(timer);
    }
  }, [finishedStroke, startedStroke, analyze]);

  return (
    <>
      <GestureHandlerRootView style={{ flex: 1 }}>
        <View
          style={{
            flex: 1,
          }}
        >
          <View style={{ flex: 1, marginVertical: 20 }}>
            {showBackground ? (
              <Image
                source={require("./dots3.png")}
                style={{
                  width: "100%",
                  height: "100%",
                  position: "absolute",
                }}
                resizeMode="repeat"
              />
            ) : (
              <></>
            )}

            <GestureDetector gesture={tap}>
              <GestureDetector gesture={pan}>
                <Canvas style={{ flex: 8 }}>
                  {paths.map((p, index) => (
                    <Path
                      key={index}
                      path={p.segments.join(" ")}
                      strokeWidth={strokeWid}
                      style="stroke"
                      color={p.color}
                    />
                  ))}

                  {circles.map((c, index) => (
                    <Circle
                      key={index}
                      cx={c.x}
                      cy={c.y}
                      r={strokeWid}
                      color={c.color}
                    />
                  ))}
                </Canvas>
              </GestureDetector>
            </GestureDetector>
          </View>

          <View style={styles.toolbar}>
            <View style={styles.stroke}>
              {showStrokes && (
                <View style={styles.widthContainer}>
                  {constants.strokes.map((stroke) => (
                    <Stroke
                      key={stroke}
                      stroke={stroke}
                      onPress={() => {
                        setShowStrokes(!showStrokes);
                        setStrokeWid(stroke);
                      }}
                    />
                  ))}
                </View>
              )}

              <View
                style={{
                  backgroundColor: "#f7f7f7",
                  borderRadius: 5,
                }}
              >
                {showStrokes && (
                  <View
                    style={{
                      width: 5,
                      height: 5,
                      borderRadius: 100,
                      backgroundColor: "black",
                      alignSelf: "center",
                      position: "absolute",
                    }}
                  />
                )}
              </View>

              <View style={{ alignSelf: "center" }}>
                <Stroke
                  stroke={strokeWid}
                  onPress={() => {
                    setShowStrokes(!showStrokes);
                  }}
                />
              </View>
            </View>

            <Separator />

            <View style={styles.colorContainer}>
              {constants.colors.map((item, i) => (
                <Color
                  key={item}
                  color={item}
                  onPress={() => {
                    setActivePaletteColorIndex(i);
                    paletteVisible.value = false;
                    setColor(item);
                  }}
                />
              ))}
            </View>

            <Separator />

            <View style={styles.buttonsContainer}>
              <TouchableOpacity onPress={undoPath}>
                <Ionicons
                  name="md-arrow-undo-outline"
                  style={styles.icon}
                ></Ionicons>
              </TouchableOpacity>

              <TouchableOpacity onPress={clearCanvas}>
                <MaterialCommunityIcons name="eraser" size={28} color="black" />
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </GestureHandlerRootView>
    </>
  );
}

const styles = StyleSheet.create({
  icon: {
    fontSize: 28,
    textAlign: "center",
  },

  colorContainer: {
    flex: 7,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-around",
  },

  toolbar: {
    backgroundColor: "#ffffff",
    height: 50,
    width: 340,
    borderRadius: 100,
    flexDirection: "row",
    paddingHorizontal: 12,
    justifyContent: "center",
    alignItems: "center",
    alignSelf: "center",
    ...utils.getElevation(5),
  },

  widthContainer: {
    flex: 1,
    bottom: 50,
    height: 40,
    backgroundColor: "#ffffff",
    position: "absolute",
    flexDirection: "row",
    alignItems: "center",
    borderRadius: 100,
    ...utils.getElevation(3),
  },

  buttonsContainer: {
    flex: 2.2,
    flexDirection: "row",
    justifyContent: "space-around",
  },

  stroke: {
    flex: 1,
  },

  canvas: {},
});
