import torch
import enum


class BoxFormat(enum.Enum):
    XYXY = "xyxy"  # [x1, y1, x2, y2]
    XYWH = "xywh"  # [x, y, width, height]


class CoordinateSystem(enum.Enum):
    ABSOLUTE = "absolute"  # Pixel coordinates
    NORMALIZED = "normalized"  # Normalized coordinates [0, 1]


class BoxRescaler:
    def __init__(self, original_size, max_image_side=1024):
        self.original_w, self.original_h = original_size
        self.max_image_side = max_image_side

        self.scale = max(self.original_w, self.original_h) / self.max_image_side
        self.new_w = int(self.original_w / self.scale)
        self.new_h = int(self.original_h / self.scale)
        self.pad_w = (self.max_image_side - self.new_w) / 2
        self.pad_h = (self.max_image_side - self.new_h) / 2

        # Calculate the dimensions of the padded image
        self.padded_w = self.new_w + 2 * self.pad_w
        self.padded_h = self.new_h + 2 * self.pad_h

    def _convert_format(self, boxes, from_format, to_format):
        if from_format == to_format:
            return boxes

        if from_format == BoxFormat.XYXY and to_format == BoxFormat.XYWH:
            x1, y1, x2, y2 = boxes.unbind(-1)
            return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)
        elif from_format == BoxFormat.XYWH and to_format == BoxFormat.XYXY:
            x, y, w, h = boxes.unbind(-1)
            return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=-1)
        else:
            raise ValueError(f"Unsupported conversion: {from_format} to {to_format}")

    def _normalize(self, boxes, from_system):
        if from_system == CoordinateSystem.NORMALIZED:
            return boxes
        x, y, w, h = boxes.unbind(-1)
        return torch.stack(
            [
                x / self.padded_w,
                y / self.padded_h,
                w / self.padded_w,
                h / self.padded_h,
            ],
            dim=-1,
        )

    def _denormalize(self, boxes, to_system):
        if to_system == CoordinateSystem.NORMALIZED:
            return boxes
        x, y, w, h = boxes.unbind(-1)
        return torch.stack(
            [
                x * self.padded_w,
                y * self.padded_h,
                w * self.padded_w,
                h * self.padded_h,
            ],
            dim=-1,
        )

    def _add_padding(self, boxes):
        x, y, w, h = boxes.unbind(-1)
        return torch.stack([x + self.pad_w, y + self.pad_h, w, h], dim=-1)

    def _remove_padding(self, boxes):
        x, y, w, h = boxes.unbind(-1)
        return torch.stack(
            [
                torch.clamp(x - self.pad_w, min=0),
                torch.clamp(y - self.pad_h, min=0),
                w,
                h,
            ],
            dim=-1,
        )

    def _scale_to_original(self, boxes):
        x, y, w, h = boxes.unbind(-1)
        return torch.stack(
            [x * self.scale, y * self.scale, w * self.scale, h * self.scale], dim=-1
        )

    def _scale_to_dino(self, boxes):
        x, y, w, h = boxes.unbind(-1)
        return torch.stack(
            [x / self.scale, y / self.scale, w / self.scale, h / self.scale], dim=-1
        )

    def _clip_to_image(self, boxes):
        x1, y1, x2, y2 = boxes.unbind(-1)
        x1 = torch.clamp(x1, min=0, max=self.original_w)
        y1 = torch.clamp(y1, min=0, max=self.original_h)
        x2 = torch.clamp(x2, min=0, max=self.original_w)
        y2 = torch.clamp(y2, min=0, max=self.original_h)
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def rescale(self, boxes, from_format, to_format, from_system, to_system):
        # Convert to XYWH format if not already
        boxes = self._convert_format(boxes, from_format, BoxFormat.XYWH)

        # Normalize if not already normalized
        boxes = self._normalize(boxes, from_system)

        if to_system == CoordinateSystem.ABSOLUTE:
            # Scale to original image size
            boxes = self._denormalize(boxes, to_system)
            boxes = self._remove_padding(boxes)
            boxes = self._scale_to_original(boxes)
        else:  # to_system == CoordinateSystem.NORMALIZED
            # Ensure boxes are in DINO format
            boxes = self._add_padding(boxes)

        # Convert to desired output format
        boxes = self._convert_format(boxes, BoxFormat.XYWH, to_format)

        # Clip to image boundaries if in absolute coordinates
        if to_system == CoordinateSystem.ABSOLUTE and to_format == BoxFormat.XYXY:
            boxes = self._clip_to_image(boxes)

        return boxes
