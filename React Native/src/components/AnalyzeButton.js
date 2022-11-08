import React from "react";
import { TouchableOpacity, View } from "react-native";
import { FontAwesome } from "@expo/vector-icons";

const AnalyzeButton = ({ pressed, setPressed }) => {
  return (
    <View>
      <TouchableOpacity onPress={() => setPressed(!pressed)}>
        <FontAwesome
          name="cogs"
          size={32}
          color={pressed ? "#525252" : "#cfcfcf"}
        />
      </TouchableOpacity>
    </View>
  );
};

export default AnalyzeButton;
