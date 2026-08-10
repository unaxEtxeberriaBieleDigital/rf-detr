import "./styles/theme.css";
import "./styles/base.css";
import "./styles/setup.css";
import "./styles/visualizer.css";
import "./styles/filterSidebar.css";
import "./styles/pcaPanel.css";
import "./styles/imageGallery.css";
import "./styles/semanticSearch.css";
import { useAppConfig } from "./context/AppContext";
import SetupPage from "./pages/SetupPage";
import VisualizerPage from "./pages/VisualizerPage";

function App() {
  const { config } = useAppConfig();

  return config ? <VisualizerPage /> : <SetupPage />;
}

export default App;
