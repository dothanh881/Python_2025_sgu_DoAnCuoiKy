import torch
import os

def check_checkpoint(checkpoint_path):
    """
    Kiểm tra thông tin trong checkpoint file
    """
    if not os.path.exists(checkpoint_path):
        print(f" Checkpoint file không tồn tại: {checkpoint_path}")
        return False
    
    try:
        print(f" Đang kiểm tra checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print("Checkpoint loaded thành công!")
        print(f" Thông tin checkpoint:")
        print(f"   - Epoch: {checkpoint['epoch']}")
        print(f"   - BLEU-4 score: {checkpoint['bleu-4']:.4f}")
        print(f"   - Epochs since improvement: {checkpoint['epochs_since_improvement']}")
        print(f"   - Epoch tiếp theo sẽ là: {checkpoint['epoch'] + 1}")
        
        # Kiểm tra các components
        components = ['encoder', 'decoder', 'encoder_optimizer', 'decoder_optimizer']
        for comp in components:
            if comp in checkpoint:
                print(f"    {comp}: OK")
            else:
                print(f"    {comp}: MISSING")
        
        return True
        
    except Exception as e:
        print(f" Lỗi khi load checkpoint: {e}")
        return False

if __name__ == "__main__":
    # Kiểm tra các checkpoint có thể có
    checkpoint_files = [
        'BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar',
        'checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar'
    ]
    
    print("🔍 KIỂM TRA CHECKPOINT FILES")
    print("=" * 50)
    
    for checkpoint_file in checkpoint_files:
        print(f"\n{checkpoint_file}:")
        check_checkpoint(checkpoint_file)
        print("-" * 30)