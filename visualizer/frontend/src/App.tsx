import "./App.css";
import { useAppConfig } from "./context/AppContext";
import SetupPage from "./pages/SetupPage";
import VisualizerPage from "./pages/VisualizerPage";

function App() {
  const { config } = useAppConfig();

  return config ? <VisualizerPage /> : <SetupPage />;
}

export default App;
