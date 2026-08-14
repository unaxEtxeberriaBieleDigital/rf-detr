import Beams from "./DefaultBackground";

export default function LoadingDiv( ) {

  return (
    <div
        style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            height: "100%",
            width: "100%",
            textAlign: "center",
        }}
    >
        <Beams
          beamWidth={3}
          beamHeight={30}
          beamNumber={20}
          lightColor="#ffffff"
          speed={2}
          noiseIntensity={1.75}
          scale={0.2}
          rotation={30}
          centralContent={<div className="loader" />}
        />
    </div>
  );
}
