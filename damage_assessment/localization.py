"""M2 damage localization: backend-driven DDM generation and DEM scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from preprocessing.lbp import LBPFeatureExtractor

from damage_assessment.localization_backends import LocalizationResult, get_localization_backend, normalize_map

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass(frozen=True)
class DamageRegion:
    """One connected damaged component with geometric confidence metadata."""

    area_pixels: int
    area_percentage: float
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    compactness: float
    boundary_complexity: float
    mean_activation: float
    region_confidence: float
    rank: int
    polygon: List[Tuple[int, int]]

    @property
    def mean_saliency(self) -> float:
        """Backward-compatible alias for older callers."""
        return self.mean_activation


@dataclass(frozen=True)
class DamageAssessmentResult:
    """Complete M2 output for dashboard and downstream use."""

    dem_score: float
    severity_level: str
    affected_area_percentage: float
    damaged_region_count: int
    largest_region_percentage: float
    average_region_size_percentage: float
    average_damage_density: float
    boundary_complexity: float
    texture_entropy: float
    localization_confidence: float
    average_activation: float
    localization_score: float
    localization_backend: str
    centroid: Tuple[float, float] | None
    regions: List[DamageRegion]
    damage_mask: np.ndarray
    ddm_overlay: np.ndarray
    boundaries: List[Tuple[int, int, int, int]]
    impact_estimates: Dict[str, float]
    emergency_recommendations: List[str]
    backend_metadata: Dict[str, object]


class DamageLocalizationAnalyzer:
    """Convert localization maps into damage regions, DDM, and DEM."""

    def __init__(
        self,
        localization_backend: str = "sun_ica",
        threshold_percentile: float | None = None,
        min_region_ratio: float = 0.0015,
    ) -> None:
        self.localization_backend = localization_backend
        self.threshold_percentile = threshold_percentile
        self.min_region_ratio = min_region_ratio
        self.lbp = LBPFeatureExtractor()

    def assess(
        self,
        img_rgb: np.ndarray,
        saliency_map: np.ndarray | None = None,
        disaster_label: str = "unknown",
        localization_backend: str | None = None,
    ) -> DamageAssessmentResult:
        backend_name = localization_backend or self.localization_backend
        loc = self._localize(img_rgb, saliency_map, backend_name)
        activation = normalize_map(loc.activation_map)
        mask = self._segment_damage(activation, disaster_label)
        regions = self._connected_regions(mask, activation)
        affected = float(mask.mean() * 100.0)
        largest = max((region.area_percentage for region in regions), default=0.0)
        avg_size = float(np.mean([region.area_percentage for region in regions])) if regions else 0.0
        density = float(activation[mask > 0].mean()) if np.any(mask) else 0.0
        avg_activation = float(activation.mean())
        compactness = float(np.mean([region.compactness for region in regions])) if regions else 0.0
        boundary_complexity = float(np.mean([region.boundary_complexity for region in regions])) if regions else 0.0
        texture_entropy = self._texture_entropy(img_rgb, mask)
        localization_confidence = float(np.clip(loc.confidence * 3.0, 0.0, 1.0))
        localization = self._localization_score(affected, density, len(regions), compactness, localization_confidence)
        dem = self._dem_score(
            affected,
            density,
            largest,
            avg_size,
            len(regions),
            compactness,
            boundary_complexity,
            texture_entropy,
            localization_confidence,
            avg_activation,
            disaster_label,
        )
        severity = self._severity_label(dem)
        centroid = self._weighted_centroid(regions)
        ddm = self._create_ddm_overlay(img_rgb, activation, mask, regions)
        estimates = self._impact_estimates(disaster_label, dem, affected, largest, density)
        recommendations = self._recommendations(disaster_label, severity, affected, backend_name)

        return DamageAssessmentResult(
            dem_score=dem,
            severity_level=severity,
            affected_area_percentage=affected,
            damaged_region_count=len(regions),
            largest_region_percentage=largest,
            average_region_size_percentage=avg_size,
            average_damage_density=density,
            boundary_complexity=boundary_complexity,
            texture_entropy=texture_entropy,
            localization_confidence=localization_confidence,
            average_activation=avg_activation,
            localization_score=localization,
            localization_backend=loc.backend,
            centroid=centroid,
            regions=regions,
            damage_mask=mask,
            ddm_overlay=ddm,
            boundaries=[region.bbox for region in regions],
            impact_estimates=estimates,
            emergency_recommendations=recommendations,
            backend_metadata=loc.metadata,
        )

    def _localize(self, img_rgb: np.ndarray, saliency_map: np.ndarray | None, backend_name: str) -> LocalizationResult:
        if saliency_map is not None and backend_name in {"provided", "saliency"}:
            activation = normalize_map(saliency_map)
            return LocalizationResult("provided_saliency", activation, float(np.std(activation)), {"source": "caller"})
        backend = get_localization_backend(backend_name)
        return backend.localize(img_rgb)

    def _threshold_for_disaster(self, disaster_label: str) -> float:
        if self.threshold_percentile is not None:
            return self.threshold_percentile
        label = disaster_label.lower()
        if "flood" in label:
            return 68.0
        if "wildfire" in label:
            return 72.0
        if "hurricane" in label:
            return 69.0
        if "earthquake" in label:
            return 70.0
        if "landslide" in label:
            return 71.0
        return 70.0

    def _segment_damage(self, activation: np.ndarray, disaster_label: str) -> np.ndarray:
        activation = normalize_map(activation)
        if cv2 is not None:
            activation = cv2.GaussianBlur(activation.astype(np.float32), (5, 5), 0)
            adaptive = cv2.adaptiveThreshold(
                np.uint8(activation * 255),
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                -2,
            )
            adaptive = (adaptive > 0).astype(np.uint8)
        else:
            adaptive = np.zeros_like(activation, dtype=np.uint8)

        percentile_threshold = np.percentile(activation, self._threshold_for_disaster(disaster_label))
        percentile_mask = (activation >= percentile_threshold).astype(np.uint8)
        mask = np.maximum(percentile_mask, adaptive)
        if cv2 is None:
            return mask

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        min_area = max(12, int(mask.size * self.min_region_ratio))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean = np.zeros_like(mask)
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            component = (labels == label_id).astype(np.uint8)
            region_activation = float(activation[labels == label_id].mean())
            if region_activation < 0.32:
                continue
            clean[component > 0] = 1
        return clean

    def _connected_regions(self, mask: np.ndarray, activation: np.ndarray) -> List[DamageRegion]:
        if cv2 is None or not np.any(mask):
            return []
        total_pixels = float(mask.size)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        raw_regions: List[DamageRegion] = []
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            component = (labels == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea) if contours else np.empty((0, 1, 2), dtype=np.int32)
            perimeter = float(cv2.arcLength(contour, True)) if contour.size else 0.0
            compactness = float(4.0 * np.pi * area / max(perimeter * perimeter, 1.0))
            boundary_complexity = float(perimeter / max(np.sqrt(area), 1.0))
            mean_activation = float(activation[labels == label_id].mean())
            area_percentage = float(area * 100.0 / total_pixels)
            region_confidence = float(np.clip(0.65 * mean_activation + 0.35 * min(area_percentage / 25.0, 1.0), 0.0, 1.0))
            polygon = self._polygon(contour)
            raw_regions.append(
                DamageRegion(
                    area_pixels=area,
                    area_percentage=area_percentage,
                    centroid=(float(centroids[label_id][0]), float(centroids[label_id][1])),
                    bbox=(x, y, w, h),
                    compactness=compactness,
                    boundary_complexity=boundary_complexity,
                    mean_activation=mean_activation,
                    region_confidence=region_confidence,
                    rank=0,
                    polygon=polygon,
                )
            )
        ranked = sorted(raw_regions, key=lambda item: item.region_confidence * item.area_pixels, reverse=True)
        return [region.__class__(**{**region.__dict__, "rank": idx + 1}) for idx, region in enumerate(ranked)]

    @staticmethod
    def _polygon(contour: np.ndarray) -> List[Tuple[int, int]]:
        if cv2 is None or contour.size == 0:
            return []
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return [(int(point[0][0]), int(point[0][1])) for point in approx[:64]]

    def _create_ddm_overlay(
        self,
        img_rgb: np.ndarray,
        activation: np.ndarray,
        mask: np.ndarray,
        regions: List[DamageRegion],
    ) -> np.ndarray:
        base = img_rgb.astype(np.uint8).copy()
        if cv2 is None:
            red = np.zeros_like(base)
            red[..., 0] = 255
            return np.where(mask[..., None] > 0, (0.55 * red + 0.45 * base).astype(np.uint8), base)
        heat = cv2.applyColorMap(np.uint8(normalize_map(activation) * 255), cv2.COLORMAP_TURBO)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(heat, 0.55, base, 0.45, 0)
        output = np.where(mask[..., None] > 0, overlay, base)
        for region in regions[:12]:
            x, y, w, h = region.bbox
            color = (255, 255, 255) if region.rank == 1 else (49, 208, 125)
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            cv2.circle(output, (int(region.centroid[0]), int(region.centroid[1])), 4, (20, 255, 180), -1)
        return output.astype(np.uint8)

    def _texture_entropy(self, img_rgb: np.ndarray, mask: np.ndarray) -> float:
        lbp = self.lbp.extract_from_rgb(img_rgb)
        texture = lbp.texture_map
        if texture.shape != mask.shape:
            if cv2 is not None:
                texture = cv2.resize(texture, (mask.shape[1], mask.shape[0]))
            else:
                return float(lbp.statistics["entropy"] / 4.0)
        if not np.any(mask):
            return float(np.clip(lbp.statistics["entropy"] / 4.0, 0.0, 1.0))
        values = texture[mask > 0]
        hist, _ = np.histogram(values, bins=16, range=(0.0, 1.0), density=False)
        hist = hist.astype(np.float32) / max(float(hist.sum()), 1.0)
        entropy = float(-np.sum(hist * np.log2(np.clip(hist, 1e-10, 1.0))) / np.log2(16))
        return float(np.clip(entropy, 0.0, 1.0))

    def _dem_score(
        self,
        affected_area_percentage: float,
        density: float,
        largest_region_percentage: float,
        average_region_size_percentage: float,
        region_count: int,
        compactness: float,
        boundary_complexity: float,
        texture_entropy: float,
        localization_confidence: float,
        average_activation: float,
        disaster_label: str,
    ) -> float:
        area_term = affected_area_percentage / 100.0
        largest_term = largest_region_percentage / 100.0
        avg_region_term = min(average_region_size_percentage / 25.0, 1.0)
        count_term = min(region_count / 8.0, 1.0)
        compact_term = np.clip(1.0 - compactness, 0.0, 1.0)
        boundary_term = np.clip(boundary_complexity / 18.0, 0.0, 1.0)
        raw = (
            0.22 * area_term
            + 0.18 * density
            + 0.13 * largest_term
            + 0.09 * avg_region_term
            + 0.10 * count_term
            + 0.08 * compact_term
            + 0.07 * boundary_term
            + 0.06 * texture_entropy
            + 0.04 * localization_confidence
            + 0.03 * average_activation
        )
        return float(np.clip(raw * self._severity_weight(disaster_label) * 100.0, 0.0, 100.0))

    @staticmethod
    def _localization_score(
        affected_area_percentage: float,
        density: float,
        region_count: int,
        compactness: float,
        confidence: float,
    ) -> float:
        score = 0.38 * density + 0.22 * min(affected_area_percentage / 45.0, 1.0)
        score += 0.17 * min(region_count / 6.0, 1.0) + 0.10 * np.clip(compactness, 0.0, 1.0)
        score += 0.13 * confidence
        return float(np.clip(score * 100.0, 0.0, 100.0))

    @staticmethod
    def _severity_weight(disaster_label: str) -> float:
        label = disaster_label.lower()
        if "earthquake" in label or "wildfire" in label:
            return 1.18
        if "flood" in label or "hurricane" in label:
            return 1.08
        if "landslide" in label:
            return 1.12
        return 1.0

    @staticmethod
    def _severity_label(score: float) -> str:
        if score <= 20:
            return "Minimal"
        if score <= 40:
            return "Mild"
        if score <= 60:
            return "Moderate"
        if score <= 80:
            return "Severe"
        return "Critical"

    @staticmethod
    def _weighted_centroid(regions: List[DamageRegion]) -> Tuple[float, float] | None:
        if not regions:
            return None
        weight_sum = float(sum(region.area_pixels for region in regions))
        x = sum(region.centroid[0] * region.area_pixels for region in regions) / weight_sum
        y = sum(region.centroid[1] * region.area_pixels for region in regions) / weight_sum
        return float(x), float(y)

    @staticmethod
    def _impact_estimates(
        disaster_label: str,
        dem: float,
        affected: float,
        largest: float,
        density: float,
    ) -> Dict[str, float]:
        label = disaster_label.lower()
        base = dem / 100.0
        estimates = {
            "infrastructure_damage": float(np.clip(0.55 * base + 0.45 * largest / 100.0, 0.0, 1.0) * 100.0),
            "vegetation_impact": float(np.clip(0.45 * base + 0.55 * density, 0.0, 1.0) * 100.0),
            "flooded_area_percentage": 0.0,
            "wildfire_spread": 0.0,
            "building_destruction_estimate": float(np.clip(0.7 * base + 0.3 * affected / 100.0, 0.0, 1.0) * 100.0),
            "disaster_severity_score": float(dem),
        }
        if "flood" in label or "hurricane" in label:
            estimates["flooded_area_percentage"] = float(np.clip(affected * (0.65 + 0.35 * density), 0.0, 100.0))
        if "wildfire" in label:
            estimates["wildfire_spread"] = float(np.clip(dem * (0.7 + 0.3 * density), 0.0, 100.0))
            estimates["vegetation_impact"] = float(np.clip(dem * 1.08, 0.0, 100.0))
        if "earthquake" in label:
            estimates["building_destruction_estimate"] = float(np.clip(dem * 1.05, 0.0, 100.0))
            estimates["infrastructure_damage"] = float(np.clip(dem * 1.1, 0.0, 100.0))
        return estimates

    @staticmethod
    def _recommendations(disaster_label: str, severity: str, affected: float, backend: str) -> List[str]:
        label = disaster_label.lower()
        recommendations = [
            f"Review {backend} DDM regions before operational deployment.",
            "Prioritize field validation of highlighted high-confidence regions.",
            "Route imagery and coordinates to the incident command damage desk.",
        ]
        if severity in {"Severe", "Critical"} or affected > 35:
            recommendations.append("Escalate to rapid response planning and resource staging.")
        if "flood" in label:
            recommendations.append("Check evacuation routes, drainage channels, and low-lying shelters.")
        elif "wildfire" in label:
            recommendations.append("Review containment lines, wind direction, and nearby vegetation corridors.")
        elif "earthquake" in label:
            recommendations.append("Inspect bridges, dense building clusters, and blocked-access corridors.")
        elif "landslide" in label:
            recommendations.append("Assess slope stability and blocked road segments near highlighted regions.")
        return recommendations
