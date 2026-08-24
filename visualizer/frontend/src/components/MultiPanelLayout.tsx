import { useState, useRef, ReactNode } from "react";
import "../styles/multiPanelLayout.css";

export interface PanelDefinition {
  id: string;
  title: string;
  closable?: boolean;
  component: () => ReactNode;
}

interface MultiPanelLayoutProps {
  /** Array of active panel IDs managed by the parent */
  activePanelIds: string[];
  /** Callback fired when a panel is added or removed */
  onActivePanelIdsChange: (ids: string[]) => void;
  /** All available panels that can be displayed */
  availablePanels: PanelDefinition[];
}

export default function MultiPanelLayout({
  activePanelIds,
  onActivePanelIdsChange,
  availablePanels,
}: MultiPanelLayoutProps) {
  const [showPanelMenu, setShowPanelMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Derivamos los paneles visibles directamente de los IDs activos proporcionados por el padre
  const visiblePanels = activePanelIds
    .map((id) => availablePanels.find((p) => p.id === id))
    .filter((p): p is PanelDefinition => p !== undefined);

  const handleAddPanel = (panelId: string) => {
    onActivePanelIdsChange([...activePanelIds, panelId]);
    setShowPanelMenu(false);
  };

  const handleRemovePanel = (panelId: string) => {
    // Nunca permitir cerrar todos los paneles - mantener al menos uno
    if (activePanelIds.length <= 1) return;
    onActivePanelIdsChange(activePanelIds.filter((id) => id !== panelId));
  };

  const panelWidth = `calc(100% / ${visiblePanels.length})`;

  // Filtrar los paneles que ya están visibles para el menú "+"
  const visiblePanelIds = new Set(activePanelIds);
  const panelsToAdd = availablePanels.filter((panel) => !visiblePanelIds.has(panel.id));

  return (
    <div className="multi-panel-layout">
      <div className="panels-top-bar">
        <div className="add-panel-container" ref={menuRef}>
          <button
            className="add-panel-button"
            onClick={() => setShowPanelMenu(!showPanelMenu)}
            title="Agregar panel"
            aria-label="Agregar panel"
          >
            +
          </button>
          {showPanelMenu && panelsToAdd.length > 0 && (
            <div className="panel-menu">
              {panelsToAdd.map((panel) => (
                <button
                  key={panel.id}
                  className="panel-menu-item"
                  onClick={() => handleAddPanel(panel.id)}
                >
                  + {panel.title}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="panels-container">
        {visiblePanels.map((panelDef) => (
          <div
            key={panelDef.id}
            className="panel-wrapper"
            style={{ width: panelWidth }}
          >
            <div className="panel-header">
              <div className="panel-tabs">
                <div className="panel-title">
                  {panelDef.title}
                </div>
              </div>
              {visiblePanels.length > 1 && panelDef.closable !== false && (
                <button
                  className="close-button"
                  onClick={() => handleRemovePanel(panelDef.id)}
                  title="Cerrar panel"
                  aria-label="Cerrar panel"
                >
                  ✕
                </button>
              )}
            </div>
            <div className="panel-content">
              {panelDef.component()}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}