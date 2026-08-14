// ------------------------------------------------------------------------
// RF-DETR
// Copyright (c) 2025 Roboflow. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 [see LICENSE for details]
// ------------------------------------------------------------------------

import type { FC } from "react";
import "../styles/segmentedControl.css";

export interface SegmentedOption<T extends string | number> {
  value: T;
  label: string;
  title?: string;
  disabled?: boolean;
}

interface SegmentedControlProps<T extends string | number> {
  options: SegmentedOption<T>[];
  value: T;
  onChange: (value: T) => void;
  disabled?: boolean;
  className?: string;
}

/** A generic pill-style segmented control that supports 2 or more options.
 *
 *  The active option is rendered with a white background and black text;
 *  inactive options use a dark background with muted text.
 */
function SegmentedControl<T extends string | number>({
  options,
  value,
  onChange,
  disabled = false,
  className = "",
}: SegmentedControlProps<T>): ReturnType<FC> {
  return (
    <div className={`seg-ctrl ${className}`.trim()}>
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={String(opt.value)}
            type="button"
            className={`seg-ctrl__btn${active ? " seg-ctrl__btn--active" : ""}`}
            onClick={() => !disabled && !opt.disabled && onChange(opt.value)}
            disabled={disabled || opt.disabled}
            title={opt.title}
            aria-pressed={active}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

export default SegmentedControl;
