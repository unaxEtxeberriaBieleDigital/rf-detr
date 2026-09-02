import "./styles/theme.css";
import "./styles/base.css";
import "./styles/setup.css";
import "./styles/visualizer.css";
import "./styles/filterSidebar.css";
import "./styles/embeddingsPlot.css";
import "./styles/imageGallery.css";
import "./styles/semanticSearch.css";
import { useAppConfig } from "./context/AppContext";
import BackendGate from "./components/BackendGate";
import SetupPage from "./pages/SetupPage";
import VisualizerPage from "./pages/VisualizerPage";

function App() {
  const { config } = useAppConfig();

  return <BackendGate>{config ? <VisualizerPage /> : <SetupPage />}</BackendGate>;
}

export default App;
