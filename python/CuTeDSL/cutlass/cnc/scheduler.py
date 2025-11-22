"""
Tag manipulation utilities for scheduling and loop transformations.

Tags represent the hierarchical coordinate space, where None indicates
an unbound/unscheduled dimension, and SymInt represents a bound loop variable.
"""

from .symbolic_tag import SymCoord, SymInt
from typing import Optional, Tuple


from .symbolic_tag import SymCoord, SymInt
from typing import Optional


def tag_push(tag: SymCoord, target: SymCoord) -> SymCoord:
    """
    Push a new target into the first available None slot in the tag.
    
    This represents binding a new coordinate to the next unscheduled dimension.
    
    Args:
        tag: Current tag (coordinate space state)
        target: New coordinate to bind (can be SymInt, tuple, or None)
        
    Returns:
        New tag with target pushed into the first None position
        
    Examples:
        tag_push(None, SymInt('i'))                    # -> SymInt('i')
        tag_push((None, None), SymInt('i'))            # -> (SymInt('i'), None)
        tag_push((None, None), (SymInt('i'), SymInt('j')))  # -> ((SymInt('i'), SymInt('j')), None)
        tag_push((SymInt('i'), None), SymInt('j'))     # -> (SymInt('i'), SymInt('j'))
    """
    # If tag itself is None, return the target
    if tag is None:
        return target
    
    # If tag is a SymInt, nowhere to push
    if isinstance(tag, SymInt):
        return tag
    
    # If tag is a tuple, find and replace first None
    if isinstance(tag, tuple):
        result = []
        replaced = False
        
        for item in tag:
            if not replaced and item is None:
                result.append(target)
                replaced = True
            elif not replaced and isinstance(item, tuple):
                new_item = tag_push(item, target)
                result.append(new_item)
                if new_item != item:
                    replaced = True
            else:
                result.append(item)
        
        return tuple(result)
    
    return tag