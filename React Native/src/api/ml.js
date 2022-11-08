import axios from "axios";

const IP = "http://172.20.10.2:5000";

export default axios.create({
  baseURL: IP,
});
