import React from "react";
import { TouchableOpacity, View, Share } from "react-native";
import { EvilIcons } from "@expo/vector-icons";

const ShareButton = ({ textToShare, noteTitle }) => {
  console.log(textToShare);
  const onShare = async () => {
    try {
      await Share.share({
        message: textToShare,
        title: noteTitle,
      });
    } catch (error) {
      alert(error.message);
    }
  };

  return (
    <View style={{ marginLeft: 10 }}>
      <TouchableOpacity onPress={onShare}>
        <EvilIcons name="share-apple" size={32} color="#969696" />
      </TouchableOpacity>
    </View>
  );
};

ShareButton.defaultProps = {
  initialValues: {
    textToShare: "No text",
    noteTitle: "No title",
  },
};

export default ShareButton;
