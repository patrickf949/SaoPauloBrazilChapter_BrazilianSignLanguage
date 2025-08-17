import { BeatLoader, DotLoader } from "react-spinners";

export default function Loader() {
  return (
    <div style={{ display: "flex", justifyContent: "center", padding: "2rem" }}>
      <BeatLoader color="#004770ff" size={10} />
    </div>
  );
}


export function VideoLoader() {
  return (
    // <div style={{ display: "flex", justifyContent: "center", padding: "2rem", opacity:0.3 }}>
      <DotLoader color="#686868ff" cssOverride={{opacity:0.2}} size={200} />
    // </div>
  );
}