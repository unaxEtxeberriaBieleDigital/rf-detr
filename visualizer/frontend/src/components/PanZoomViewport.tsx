import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";

export interface PanZoomHandle {
  reset: () => void;
}

interface PanZoomViewportProps {
  children: React.ReactNode;
  /** Whenever this value changes, scale/offset are reset back to their defaults. */
  resetKey?: string | number | null;
}

/** Wheel-zoom + drag-pan viewport, shared by the image viewer and the semantic-search panel. */
const PanZoomViewport = forwardRef<PanZoomHandle, PanZoomViewportProps>(function PanZoomViewport(
  { children, resetKey },
  ref,
) {
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef({
    active: false,
    pointerId: -1,
    startX: 0,
    startY: 0,
    originX: 0,
    originY: 0,
  });

  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);

  function reset(): void {
    setScale(1);
    setOffset({ x: 0, y: 0 });
  }

  useImperativeHandle(ref, () => ({ reset }));

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => reset(), [resetKey]);

  function clampScale(value: number): number {
    return Math.min(8, Math.max(0.5, value));
  }

  return (
    <div
      ref={viewportRef}
      className={`image-viewer-viewport ${isDragging ? "dragging" : ""}`}
      onWheel={(event) => {
        event.preventDefault();
        const viewport = viewportRef.current;
        if (!viewport) return;

        const rect = viewport.getBoundingClientRect();
        const pointerX = event.clientX - rect.left;
        const pointerY = event.clientY - rect.top;
        const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
        const nextScale = clampScale(scale * zoomFactor);
        if (nextScale === scale) return;

        const worldX = (pointerX - offset.x) / scale;
        const worldY = (pointerY - offset.y) / scale;
        setOffset({
          x: pointerX - worldX * nextScale,
          y: pointerY - worldY * nextScale,
        });
        setScale(nextScale);
      }}
      onPointerDown={(event) => {
        if (event.button !== 0) return;
        const target = event.currentTarget;
        dragRef.current = {
          active: true,
          pointerId: event.pointerId,
          startX: event.clientX,
          startY: event.clientY,
          originX: offset.x,
          originY: offset.y,
        };
        target.setPointerCapture(event.pointerId);
        setIsDragging(true);
      }}
      onPointerMove={(event) => {
        if (!dragRef.current.active) return;
        const deltaX = event.clientX - dragRef.current.startX;
        const deltaY = event.clientY - dragRef.current.startY;
        setOffset({
          x: dragRef.current.originX + deltaX,
          y: dragRef.current.originY + deltaY,
        });
      }}
      onPointerUp={(event) => {
        if (dragRef.current.active && event.currentTarget.hasPointerCapture(dragRef.current.pointerId)) {
          event.currentTarget.releasePointerCapture(dragRef.current.pointerId);
        }
        dragRef.current.active = false;
        setIsDragging(false);
      }}
      onPointerLeave={() => {
        if (!dragRef.current.active) return;
        dragRef.current.active = false;
        setIsDragging(false);
      }}
    >
      <div
        className="image-viewer-stage"
        style={{ transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})` }}
      >
        {children}
      </div>
    </div>
  );
});

export default PanZoomViewport;
