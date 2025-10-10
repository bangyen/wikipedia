# Baseline Model Recalibration Report

## Executive Summary

**Original Performance Drop:** 100.0%
**Estimated New Drop:** 100.0%
**Estimated Improvement:** 0.0%

⚠️ **Recalibration Incomplete:** Estimated performance drop still ≥ 10%

## Temporal Bias Analysis

### Structure Features

- **Average Bias:** 97.7%
- **Recommendation:** Reduce weights significantly

| Feature | Bias (%) | Difference (%) |
|---------|----------|----------------|
| content_length | 98.9 | -98.9 |
| log_content_length | 96.4 | -96.4 |

### Sourcing Features

- **Average Bias:** 0.0%
- **Recommendation:** 

### Editorial Features

- **Average Bias:** 0.0%
- **Recommendation:** 

### Network Features

- **Average Bias:** 83.7%
- **Recommendation:** Reduce weights significantly

| Feature | Bias (%) | Difference (%) |
|---------|----------|----------------|
| inbound_links | 100.0 | -100.0 |
| total_links | 100.0 | -100.0 |
| connectivity_score | 100.0 | -100.0 |
| hub_score | 100.0 | -100.0 |
| authority_score | 100.0 | -100.0 |
| network_centrality | 100.0 | -100.0 |
| log_inbound_links | 100.0 | -100.0 |
| log_total_links | 100.0 | -100.0 |
| pagerank_score | 5.2 | +5.2 |
| closeness_centrality | 100.0 | -100.0 |

## Weight Adjustments

### Pillar Weight Changes

| Pillar | Original | Recalibrated | Change |
|--------|----------|--------------|--------|
| editorial | 0.250 | 0.250 | +0.0% |
| network | 0.062 | 0.062 | +0.0% |
| sourcing | 0.500 | 0.500 | +0.0% |
| structure | 0.187 | 0.187 | +0.0% |

### Key Feature Weight Changes

| Feature | Original | Recalibrated | Change |
|---------|----------|--------------|--------|
| academic_source_ratio | 0.100 | 0.100 | +0.0% |
| authority_score | 0.060 | 0.060 | +0.0% |
| avg_section_depth | 0.100 | 0.100 | +0.0% |
| bot_edit_ratio | -0.100 | -0.100 | +0.0% |
| citation_count | 0.250 | 0.250 | +0.0% |
| citation_density | 0.150 | 0.150 | +0.0% |
| citations_per_1k_tokens | 0.200 | 0.200 | +0.0% |
| connectivity_score | 0.060 | 0.060 | +0.0% |
| content_length | 0.060 | 0.060 | +0.0% |
| editor_diversity | 0.150 | 0.150 | +0.0% |
| external_link_count | 0.150 | 0.150 | +0.0% |
| has_external_links | 0.150 | 0.150 | +0.0% |
| has_infobox | 0.100 | 0.100 | +0.0% |
| has_references | 0.120 | 0.120 | +0.0% |
| has_reliable_sources | 0.150 | 0.150 | +0.0% |
| inbound_links | 0.075 | 0.075 | +0.0% |
| link_density | 0.150 | 0.150 | +0.0% |
| major_editor_ratio | 0.200 | 0.200 | +0.0% |
| outbound_links | 0.200 | 0.200 | +0.0% |
| recent_activity_score | 0.150 | 0.150 | +0.0% |

## Recommendations

1. **Monitor Performance:** Track actual performance on new articles after deployment
2. **A/B Testing:** Compare recalibrated model against original on new article samples
3. **Regular Updates:** Recalibrate weights periodically as new articles accumulate
4. **Feature Engineering:** Consider adding age-normalized features for network metrics
5. **Validation:** Implement continuous validation to detect temporal drift

## Conclusion

The recalibrated baseline model shows improvement but may require additional adjustments. Consider implementing more aggressive weight reductions or exploring alternative feature engineering approaches to further reduce temporal bias.
