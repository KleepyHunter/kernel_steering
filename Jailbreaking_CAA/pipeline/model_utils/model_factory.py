from pipeline.model_utils.model_base import ModelBase

def _is_chat_model(model_path: str) -> bool:
    """Check if the model is a chat/instruct model based on its name."""
    model_name_lower = model_path.lower()
    chat_indicators = ['chat', 'instruct', '-it', '_it', 'sft', 'rlhf', 'dpo']
    return any(indicator in model_name_lower for indicator in chat_indicators)

def construct_model_base(model_path: str, force_base: bool = False) -> ModelBase:
    """
    Construct the appropriate model class based on model path.
    
    Args:
        model_path: Path or HuggingFace model name
        force_base: If True, always use the generic BaseModel (no chat template)
    """
    # If force_base, use the generic base model
    if force_base or not _is_chat_model(model_path):
        from pipeline.model_utils.base_model import BaseModel
        return BaseModel(model_path)

    if 'qwen' in model_path.lower():
        from pipeline.model_utils.qwen_model import QwenModel
        return QwenModel(model_path)
    if 'llama-3' in model_path.lower():
        from pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model(model_path)
    elif 'gemma' in model_path.lower():
        from pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel(model_path)
    else:
        # Fallback to generic base model for unknown chat models
        from pipeline.model_utils.base_model import BaseModel
        return BaseModel(model_path)
