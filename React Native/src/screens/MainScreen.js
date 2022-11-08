import React, { useState, useEffect } from "react";
import {
  View,
  StyleSheet,
  SafeAreaView,
  Keyboard,
  Dimensions,
  TouchableOpacity,
  Button,
} from "react-native";
import { Input } from "react-native-elements";
import { Button as ThemedButton } from "@rneui/themed";
import Canv from "../../Canv";
import ShareButton from "../components/ShareButton";
import AnalyzeButton from "../components/AnalyzeButton";
import { FontAwesome5 } from "@expo/vector-icons";
import { AntDesign } from "@expo/vector-icons";

const MainScreen = ({ navigation }) => {
  const [sentence, setSentence] = useState("");
  const [analyzeOn, setAnalyzeOn] = useState(false);
  const windowHeight = Dimensions.get("window").height;

  useEffect(() => {
    navigation.setOptions({
      headerRight: () => (
        <View style={styles.headerButtons}>
          <AnalyzeButton pressed={analyzeOn} setPressed={setAnalyzeOn} />
          <ShareButton textToShare={sentence} noteTitle={"Note 1"} />
        </View>
      ),

      headerLeft: () => (
        <TouchableOpacity
          style={{
            marginHorizontal: 15,
            alignItems: "center",
            justifyContent: "center",
          }}
          onPress={navigation.toggleDrawer}
        >
          <FontAwesome5 name="bars" size={28} color="#969696" />
        </TouchableOpacity>
      ),
    });
  }, [sentence, analyzeOn]);

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: "white" }}>
      <TouchableOpacity
        activeOpacity={1}
        style={{ height: 0.42 * windowHeight }}
        onPress={Keyboard.dismiss}
      >
        <View style={{ flex: 0.9, margin: 10, paddingTop: 10 }}>
          <TouchableOpacity></TouchableOpacity>

          <Input
            value={sentence}
            placeholder={"Draw or type something..."}
            multiline
            onChangeText={(text) => setSentence(text)}
            rightIcon={() =>
              sentence ? (
                <TouchableOpacity onPress={() => setSentence("")}>
                  <AntDesign name="closecircleo" size={20} color="#808080" />
                </TouchableOpacity>
              ) : (
                <></>
              )
            }
          />
        </View>
      </TouchableOpacity>

      <View style={{ flex: 1, marginBottom: 40, alignItems: "center" }}>
        <View style={{ flexDirection: "row" }}>
          <ThemedButton
            title="space"
            onPress={() => setSentence(sentence + " ")}
            titleStyle={{ color: "black" }}
            containerStyle={{
              width: 200,
            }}
            radius={7}
            color={"#f2f2f2"}
          />
        </View>
        <Canv getter={sentence} setter={setSentence} analyze={analyzeOn} />
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  headerButtons: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginHorizontal: 10,
  },

  clearButton: {},
  icon: {
    fontSize: 28,
    textAlign: "center",
  },
});

export default MainScreen;
