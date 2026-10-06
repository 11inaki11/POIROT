"""
POIROT Voting System - Weighted Voting with Hamming Distance
==============================================================
This module implements a weighted voting system for POIROT protocol where each
agent's vote has a weight between 0.5 and 1.0 based on the proximity between
their position in the system and the error location they're voting for.

Similarity Metric: Hamming Normalized Inverted Distance
- Measures how many positions differ between agent's position and their vote
- Normalizes by total vector length
- Inverts to get similarity (closer = higher weight)
- Maps to [0.5, 1.0] range

Author: CORTEX Team
Date: November 11, 2025
"""

import json
import os
from typing import List, Dict, Any, Tuple, Optional


def calculate_hamming_similarity(vector_a: List[int], vector_b: List[int]) -> float:
    """
    Calculate normalized Hamming similarity between two binary vectors.
    
    Hamming distance counts positions where vectors differ.
    Similarity = 1 - (hamming_distance / vector_length)
    
    Args:
        vector_a: First binary vector [0/1, 0/1, ...]
        vector_b: Second binary vector [0/1, 0/1, ...]
    
    Returns:
        float: Similarity value between 0.0 (completely different) and 1.0 (identical)
    
    Example:
        calculate_hamming_similarity([0, 1, 0, 0], [1, 0, 0, 0])
        -> hamming_distance = 2 (positions 0 and 1 differ)
        -> similarity = 1 - (2/4) = 0.5
    """
    if len(vector_a) != len(vector_b):
        # Handle mismatch gracefully if possible, or raise error
        # For robustness, we'll try to pad or truncate if lengths differ slightly, 
        # but raising error is safer for strict validation
        if not vector_a or not vector_b:
             return 0.0
        # raise ValueError(f"Vectors must have same length: {len(vector_a)} vs {len(vector_b)}")
        # Fallback: use min length
        min_len = min(len(vector_a), len(vector_b))
        vector_a = vector_a[:min_len]
        vector_b = vector_b[:min_len]
    
    if len(vector_a) == 0:
        return 1.0  # Empty vectors are considered identical
    
    # Count positions where vectors differ
    hamming_distance = sum(a != b for a, b in zip(vector_a, vector_b))
    
    # Normalize by vector length and invert to get similarity
    similarity = 1.0 - (hamming_distance / len(vector_a))
    
    return similarity


def calculate_vote_weight(
    agent_position: List[int], 
    voted_location: List[int],
    baseline: Optional[float] = None
) -> float:
    """
    Calculate the weight of a vote based on proximity between agent and voted location.
    
    Weight formula: baseline + 0.5 × similarity
    - Minimum weight: baseline (when similarity = 0, completely different)
    - Maximum weight: baseline + 0.5 (when similarity = 1, agent votes for themselves)
    
    Args:
        agent_position: Binary vector representing agent's position in system
        voted_location: Binary vector representing where agent thinks error is
        baseline: Dynamic baseline from trust system (default: 0.25 for neutral)
    
    Returns:
        float: Vote weight between baseline and (baseline + 0.5)
    
    Examples (with baseline=0.25):
        Agent votes for themselves: [0,1,0] → [0,1,0] → weight = 0.75
        Agent votes for neighbor: [0,1,0] → [0,1,1] → weight = 0.58
        Agent votes far away: [0,1,0] → [1,0,0] → weight = 0.42
    """
    if baseline is None:
        baseline = 0.25  # Neutral baseline (P(success) = 0.5)
    
    similarity = calculate_hamming_similarity(agent_position, voted_location)
    weight = baseline + 0.5 * similarity
    return weight


def find_agent_position(agent_name: str, error_regions: List[Dict]) -> List[int]:
    """
    Find the position vector for an agent based on error_regions definition.
    
    Args:
        agent_name: Name of the agent (e.g., "Diagnosis Doctor")
        error_regions: List of error region definitions from POIROT preanalysis
    
    Returns:
        List[int]: Binary vector with 1 at agent's position, 0s elsewhere
    
    Raises:
        ValueError: If agent name not found in error_regions
    """
    # Exact matches only: first against region ids (Phase I is required to use the
    # agent ids), then against region names. Substring matching is not used because
    # it can bind an agent to the wrong region (e.g. "physio" -> "physio_data_stream").
    norm_name = agent_name.lower().strip()

    for key in ('id', 'name'):
        for i, region in enumerate(error_regions):
            if str(region.get(key, '')).lower().strip() == norm_name:
                position = [0] * len(error_regions)
                position[i] = 1
                return position

    available = [r.get('id', '?') for r in error_regions]
    raise ValueError(
        f"Agent '{agent_name}' has no region in the hazard space (region ids: {available}). "
        "Its vote cannot be weighted."
    )


def vector_to_string(vector: List[int]) -> str:
    """Convert vector to string representation for use as dict key."""
    return str(vector)


def accumulate_votes_by_location(
    agent_votes: List[Dict],
    error_regions: List[Dict]
) -> Dict[str, Dict[str, Any]]:
    """
    Accumulate weighted votes for each unique location.
    
    NUEVA LÓGICA (Multi-componente):
    - Los votos se agrupan por vector EXACTO (no por dimensión individual)
    - Si múltiples agentes votan el mismo vector, sus pesos se SUMAN
    - Soporta vectores single-component [0,1,0,0,0,0,0] y multi-component [0,1,1,0,0,0,0]
    
    Args:
        agent_votes: List of vote details with weights
        error_regions: List of error region definitions
    
    Returns:
        Dict mapping location vector string to accumulated score and details
    """
    location_scores = {}
    
    # Initialize all single-component locations with zero score
    for i, region in enumerate(error_regions):
        location = [0] * len(error_regions)
        location[i] = 1
        location_key = vector_to_string(location)
        location_scores[location_key] = {
            "name": region['name'],
            "vector": location,
            "total_score": 0.0,
            "votes_received": [],
            "is_multi_component": False
        }
    
    # NEW APPROACH: Group votes by exact vector
    # If multiple agents vote for the same vector, sum their weights
    for vote_detail in agent_votes:
        voted_location = vote_detail['voted_location']
        vote_weight = vote_detail['vote_weight']
        agent_name = vote_detail['agent_name']
        
        # Count how many 1s in the vector (multi-component detection)
        num_components = sum(voted_location)
        is_multi = num_components > 1
        
        # Create location key for this EXACT vector
        location_key = vector_to_string(voted_location)
        
        # Initialize if this vector hasn't been seen yet
        if location_key not in location_scores:
            # Build component name for multi-component vectors
            if is_multi:
                component_names = []
                for i, val in enumerate(voted_location):
                    if val == 1:
                        component_names.append(error_regions[i]['name'])
                combined_name = " + ".join(component_names)
            else:
                # Single component
                try:
                    idx = voted_location.index(1)
                    combined_name = error_regions[idx]['name']
                except ValueError:
                    combined_name = "Unknown/Zero Vector"
            
            location_scores[location_key] = {
                "name": combined_name,
                "vector": voted_location,
                "total_score": 0.0,
                "votes_received": [],
                "is_multi_component": is_multi
            }
        
        # Accumulate weight for this EXACT vector
        location_scores[location_key]['total_score'] += vote_weight
        location_scores[location_key]['votes_received'].append(agent_name)
    
    return location_scores


def calculate_dimension_probabilities(
    location_scores: Dict[str, Dict],
    error_regions: List[Dict]
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate probability distribution across all dimensions (error regions).
    
    Each dimension represents a potential error source. The probability
    indicates how likely the error originated from that dimension based
    on weighted voting.
    
    Probability = (dimension_score / total_score) × 100%
    
    Args:
        location_scores: Accumulated scores per location
        error_regions: List of error region definitions
    
    Returns:
        Dict mapping dimension ID to probability and details
    """
    total_weight = sum(loc['total_score'] for loc in location_scores.values())
    
    dimension_probs = {}
    for i, region in enumerate(error_regions):
        location = [0] * len(error_regions)
        location[i] = 1
        location_key = vector_to_string(location)
        
        # Get score for this specific single-component location
        # Note: This ignores multi-component votes for individual probability calculation
        # A better approach might be to distribute multi-component scores, but for now
        # we stick to the original logic or simple accumulation
        
        # Improved logic: Sum up scores from ALL locations that include this dimension
        score = 0.0
        for loc_data in location_scores.values():
            vec = loc_data['vector']
            if i < len(vec) and vec[i] == 1:
                # If it's a multi-component vote, we might want to split the weight
                # or count it fully. Counting fully is simpler for "involvement" probability.
                score += loc_data['total_score']
        
        # Re-calculate total weight based on this new summation method to ensure percentages make sense
        # Or keep original total_weight (sum of all unique votes)
        
        # Let's stick to the simpler version from original code if possible, 
        # but the original code seemed to only look at single-component keys.
        # Let's use the robust approach:
        
        probability = score / total_weight if total_weight > 0 else 0.0
        
        dim_id = f"{region['id']}_{region['name'].replace(' ', '_')}"
        dimension_probs[dim_id] = {
            "name": region['name'],
            "dimension_index": i,
            "total_weight": round(score, 4),
            "probability": round(probability, 4),
            "percentage": f"{round(probability * 100, 2)}%",
            "description": region.get('description', ''),
            "type": region.get('type', 'unknown')
        }
    
    return dimension_probs


def weighted_voting_analysis(
    agent_outputs: List[Dict[str, Any]],
    poirot_preanalysis: Dict[str, Any],
    trust_system: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Perform weighted voting analysis on POIROT agent outputs.
    
    Each agent's vote is weighted based on proximity between their position
    and the error location they're voting for, using Hamming normalized
    inverted distance.
    
    Args:
        agent_outputs: List of agent output JSONs, each containing:
            - agent_name: str (must match name in error_regions)
            - hazard_vector: str (description of hazard)
            - location: List[int] (binary vector of error location)
            - justification: str (reasoning)
        
        poirot_preanalysis: POIROT preanalysis JSON containing:
            - system_name: str
            - error_regions: List[Dict] with id, name, type, description
            - error_vector_example: List[int]
    
    Returns:
        Dict containing complete voting analysis
    """
    error_regions = poirot_preanalysis['error_regions']
    vector_length = len(error_regions)
    
    # Process each agent's vote
    agent_vote_details = []
    for agent_output in agent_outputs:
        agent_name = agent_output['agent_name']
        voted_location = agent_output['location']
        hazard_vector = agent_output['hazard_vector']
        
        # Validate voted_location length
        if len(voted_location) != vector_length:
            # Pad or truncate
            if len(voted_location) < vector_length:
                voted_location = voted_location + [0] * (vector_length - len(voted_location))
            else:
                voted_location = voted_location[:vector_length]
        
        # Find agent's position in the system
        agent_position = find_agent_position(agent_name, error_regions)
        
        # Get dynamic baseline from trust system (if available)
        baseline = None
        p_success = None
        if trust_system is not None:
            # Assuming trust_system has these methods
            try:
                baseline = trust_system.get_baseline(agent_name)
                p_success = trust_system.get_p_success(agent_name)
            except:
                pass
        
        # Calculate vote weight based on proximity
        similarity = calculate_hamming_similarity(agent_position, voted_location)
        vote_weight = calculate_vote_weight(agent_position, voted_location, baseline)
        
        agent_vote_details.append({
            "agent_name": agent_name,
            "agent_position": agent_position,
            "voted_location": voted_location,
            "hazard_vector": hazard_vector,
            "similarity_to_self": round(similarity, 4),
            "vote_weight": round(vote_weight, 4),
            "p_success": round(p_success, 4) if p_success is not None else None,
            "baseline": round(baseline, 4) if baseline is not None else None
        })
    
    # Accumulate votes by location (NEW: groups identical vectors and sums weights)
    location_scores = accumulate_votes_by_location(agent_vote_details, error_regions)
    
    # Calculate dimension probabilities (for visualization)
    dimension_probs = calculate_dimension_probabilities(location_scores, error_regions)
    
    # NUEVO MÉTODO: Calcular scores por dimensión individual (para comparación con GT)
    # Cada dimensión acumula el peso de todos los votos que la incluyen
    dimension_scores = [0.0] * len(error_regions)
    
    for vote_detail in agent_vote_details:
        voted_location = vote_detail['voted_location']
        vote_weight = vote_detail['vote_weight']
        
        # Para cada dimensión con 1 en el voto, sumar el peso
        for i, value in enumerate(voted_location):
            if value == 1:
                dimension_scores[i] += vote_weight
    
    # Crear el vector de scores continuo (no binario)
    # Ejemplo: [0.1, 1.6, 3.2, 0.0, 0.3] donde cada valor es la suma de pesos
    score_vector = [round(score, 4) for score in dimension_scores]
    
    # Encontrar las N dimensiones con mayor score (donde N = número de errores esperados)
    # Para esto, ordenamos por score descendente
    dimension_rankings = sorted(
        enumerate(dimension_scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Construir vector predicho (para compatibilidad con el sistema actual)
    # Usa la dimensión con mayor score como "ganadora"
    max_score_idx = dimension_rankings[0][0] if dimension_rankings else 0
    predicted_vector = [0] * len(error_regions)
    predicted_vector[max_score_idx] = 1
    
    # Información sobre empates
    total_votes = sum(dimension_scores)
    max_score = dimension_rankings[0][1] if dimension_rankings else 0
    
    # Encontrar todas las dimensiones con score máximo (empate)
    tied_dimensions = [
        {
            "dimension_index": idx,
            "name": error_regions[idx]['name'],
            "score": score,
            "percentage": round((score / total_votes * 100) if total_votes > 0 else 0.0, 2)
        }
        for idx, score in dimension_rankings
        if score == max_score and max_score > 0
    ]
    
    is_tie = len(tied_dimensions) > 1
    
    # Construir winning_location para compatibilidad
    if tied_dimensions:
        winning_location = {
            "vector": predicted_vector,
            "name": tied_dimensions[0]['name'],
            "total_score": round(max_score, 4),
            "percentage": tied_dimensions[0]['percentage'],
            "dimension_index": tied_dimensions[0]['dimension_index']
        }
    else:
        winning_location = {
            "vector": [0] * len(error_regions),
            "name": "Unknown",
            "total_score": 0.0,
            "percentage": 0.0,
            "dimension_index": -1
        }
    
    # Build final result con NUEVO FORMATO
    result = {
        "voting_summary": {
            "total_voters": len(agent_outputs),
            "total_weighted_votes": round(total_votes, 4),
            "similarity_metric": "hamming_normalized_inverted",
            "weight_range": [0.5, 1.0],
            "is_tie": is_tie,
            "tied_count": len(tied_dimensions)
        },
        "winning_location": winning_location,
        "tied_locations": tied_dimensions if is_tie else None,
        "agent_votes": agent_vote_details,
        # NUEVO: Score vector continuo por dimensión
        "dimension_scores": score_vector,
        "dimension_rankings": [
            {
                "dimension_index": idx,
                "name": error_regions[idx]['name'],
                "score": round(score, 4),
                "rank": rank + 1,
                "percentage": round((score / total_votes * 100) if total_votes > 0 else 0.0, 2)
            }
            for rank, (idx, score) in enumerate(dimension_rankings)
        ],
        "location_scores": {
            k: {
                "name": v['name'],
                "vector": v['vector'],
                "total_score": round(v['total_score'], 4),
                "percentage": round((v['total_score'] / total_votes * 100) if total_votes > 0 else 0.0, 2),
                "votes_received": v['votes_received'],
                "is_multi_component": v.get('is_multi_component', False)
            }
            for k, v in location_scores.items()
        },
        "dimension_probabilities": dimension_probs
    }
    
    return result


def print_detailed_voting_results(voting_results: Dict[str, Any], error_regions: List[Dict]):
    """
    Print a detailed breakdown of the voting results to the console.
    Matches the format requested by the user.
    """
    print("=" * 80)
    print("📊 VOTING RESULTS")
    print("=" * 80)

    # 1. Winning Location
    winner = voting_results.get('winning_location', {})
    print("\n🏆 WINNING LOCATION:")
    print(f"   Component: {winner.get('name', 'Unknown')}")
    
    # Find dimension ID if possible
    dim_idx = winner.get('dimension_index', -1)
    dim_id = error_regions[dim_idx]['id'] if 0 <= dim_idx < len(error_regions) else "?"
    
    print(f"   Dimension: {dim_id}")
    print(f"   Total Score: {winner.get('total_score', 0):.4f}")
    print(f"   Percentage: {winner.get('percentage', 0)}%")

    # 2. Individual Agent Votes
    print("\n📋 INDIVIDUAL AGENT VOTES:")
    
    agent_votes = voting_results.get('agent_votes', [])
    for vote in agent_votes:
        print(f"\n   🔹 {vote['agent_name']}:")
        print(f"      Voted for: {vote['voted_location']}")
        print(f"      Hazard identified: {vote['hazard_vector']}")
        print(f"      Vote weight: {vote['vote_weight']:.4f}")
        print(f"      Similarity to own position: {vote['similarity_to_self']:.4f}")

    # 3. Probability by Error Dimension
    print("\n" + "=" * 80)
    print("📈 PROBABILITY BY ERROR DIMENSION")
    print("=" * 80)
    print("\nEach dimension represents a potential error source.")
    print("Probabilities show likelihood that error originated there:\n")

    rankings = voting_results.get('dimension_rankings', [])
    
    # Sort by rank (should already be sorted, but just in case)
    rankings.sort(key=lambda x: x['rank'])
    
    for rank in rankings:
        idx = rank['dimension_index']
        name = rank['name']
        score = rank['score']
        percentage = rank['percentage']
        
        # Get ID
        region_id = error_regions[idx]['id'] if 0 <= idx < len(error_regions) else "?"
        
        # Create bar
        bar_length = 50
        filled_length = int(bar_length * percentage / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        print(f"\n📍 {name} ({region_id})")
        print(f"   {bar} {percentage}%")
        print(f"   Weighted votes: {score:.4f}")

    total_prob = sum(r['percentage'] for r in rankings)
    print(f"\n✓ Total probability: {total_prob:.1f}% (should be ~100%)")
    print("\n" + "=" * 80)
