/*
This file contains the API calls for the interprete service.
It uses axios to make HTTP requests to the backend service.
*/
import axios from "axios";
import { API_URL } from "@/constants";

const getInterpretation = async (data) => {
  const response = await axios.post(`${API_URL}/predict/`, data, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response;
};

export { getInterpretation };
