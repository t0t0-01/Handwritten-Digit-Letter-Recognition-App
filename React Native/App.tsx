import React, { useState } from "react";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import {
  createDrawerNavigator,
  DrawerContentScrollView,
  DrawerItemList,
} from "@react-navigation/drawer";
import { NavigationContainer } from "@react-navigation/native";
import MainScreen from "./src/screens/MainScreen";
import { Text, View, TouchableOpacity } from "react-native";
import { AntDesign } from "@expo/vector-icons";
import { Octicons } from "@expo/vector-icons";

const Stack = createNativeStackNavigator();

const Drawer = createDrawerNavigator();

const CustomDrawerContent = (props) => {
  const list = props.list;
  const setList = props.setList;

  return (
    <DrawerContentScrollView {...props}>
      <View
        style={{
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "space-between",
          padding: 15,
        }}
      >
        <Text style={{ fontSize: 25, fontWeight: "bold" }}>My Notes</Text>
        <TouchableOpacity
          onPress={() => setList([...list, list[list.length - 1] + 1])}
        >
          <AntDesign name="pluscircleo" size={24} color="black" />
        </TouchableOpacity>
      </View>
      <DrawerItemList {...props} />
    </DrawerContentScrollView>
  );
};

const App = () => {
  const [list, setList] = useState([1, 2, 3]);

  return (
    <NavigationContainer>
      <Drawer.Navigator
        screenOptions={{
          swipeEnabled: false,
          drawerIcon: ({ focused, size }) => (
            <TouchableOpacity style={{ marginRight: -15 }}>
              <Octicons name="pencil" size={24} color="black" />
            </TouchableOpacity>
          ),
        }}
        drawerContent={(props) => (
          <CustomDrawerContent {...props} list={list} setList={setList} />
        )}
      >
        {list.map((index) => (
          <Drawer.Screen
            name={"Note " + index}
            component={MainScreen}
            options={({ route, navigation }) => ({ title: "Note " + index })}
          />
        ))}
      </Drawer.Navigator>
    </NavigationContainer>
  );
};

export default App;
