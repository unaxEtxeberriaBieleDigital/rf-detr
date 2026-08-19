import { useState, useRef, ReactNode, useEffect } from "react";
import "../styles/multiPanelLayout.css";

export interface PanelDefinition {
  id: string;
  title: string;
  component: (onClose: () => void) => ReactNode;
}

interface VisiblePanel {
  panelId: string;
  panelDef: PanelDefinition;
}

interface MultiPanelLayoutProps {
  /** Initial visible panels. Typically starts with just the default (ImageGallery). */
  initialVisiblePanels: PanelDefinition[];
  /** All available panels that can be added via the "+" button. */
  availablePanels: PanelDefinition[];
}

export default function MultiPanelLayout({
  initialVisiblePanels,
  availablePanels,
}: MultiPanelLayoutProps) {
  const [visiblePanels, setVisiblePanels] = useState<VisiblePanel[]>(
    initialVisiblePanels.map((panel) => ({
      panelId: `${panel.id}-${Date.now()}`,
      panelDef: panel,
    }))
  );

  const [showPanelMenu, setShowPanelMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // When panel definitions change (e.g., filter updates), sync the component callbacks
  // for already-visible panels to get fresh props/data
  useEffect(() => {
    setVisiblePanels((current) => {
      return current.map((visible) => {
        const updated = availablePanels.find((p) => p.id === visible.panelDef.id);
        if (updated && updated !== visible.panelDef) {
          return { ...visible, panelDef: updated };
        }
        return visible;
      });
    });
  }, [availablePanels]);

  const handleAddPanel = (panelDef: PanelDefinition) => {
    const newPanel: VisiblePanel = {
      panelId: `${panelDef.id}-${Date.now()}`,
      panelDef,
    };
    setVisiblePanels([...visiblePanels, newPanel]);
    setShowPanelMenu(false);
  };

  const handleRemovePanel = (panelId: string) => {
    // Never allow removing all panels - keep at least one
    if (visiblePanels.length <= 1) return;
    setVisiblePanels(visiblePanels.filter((p) => p.panelId !== panelId));
  };

  const panelWidth = `calc(100% / ${visiblePanels.length})`;

  // Filter out panels that are already visible
  const visiblePanelIds = new Set(visiblePanels.map((p) => p.panelDef.id));
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
                  onClick={() => handleAddPanel(panel)}
                >
                  + {panel.title}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="panels-container">
        {visiblePanels.map((panel) => (
          <div
            key={panel.panelId}
            className="panel-wrapper"
            style={{ width: panelWidth }}
          >
            <div className="panel-header">
              <div className="panel-tabs">
                <div className="panel-title">
                  {panel.panelDef.title}
                </div>
              </div>
              {visiblePanels.length > 1 && (
                <button
                  className="close-button"
                  onClick={() => handleRemovePanel(panel.panelId)}
                  title="Cerrar panel"
                  aria-label="Cerrar panel"
                >
                  ✕
                </button>
              )}
            </div>
            <div className="panel-content">
              {panel.panelDef.component(() => handleRemovePanel(panel.panelId))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
