import React, { useState, useEffect } from "react";
import ML from "../api/ml";

const useML = () => {
  const [letter, setLetter] = useState("");

  const getChar = async () => {
    try {
      const response = await ML.get("/");
      setLetter(response.data);
    } catch (error) {
      console.log(error);
    }
  };

  useEffect(() => {
    getChar;
  }, []);

  return [getChar, letter];
};

export default useML;
