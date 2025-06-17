# physics_in_llm_unlearning

## Install the environment
```
conda env create -f environment.yml
```


Suppose `save_folder` is the folder for saving necessary checkpoints and results.

## Useful scripts for unlearning on (Llama2-7B, Eval-DU+)

### Finetune Models
Here is the script to finetune Llama2-7b on four types of datasets `FT-Single`, `FT-Mul`, `FT-Mul-Chunk`, `FT-Mul-Chunk-Iso`.
```
#FT-Single
bash scripts/finetune.sh 0,1 61000 llama2-7b 5 1e-05 4 FT-Single $save_folder

#FT-Mul
bash scripts/finetune.sh 0,1 61000 llama2-7b 5 1e-05 4 FT-Mul $save_folder

#FT-Mul-Chunk
bash scripts/finetune.sh 0,1 61000 llama2-7b 4 1e-05 4 FT-Mul-Chunk $save_folder

#FT-Mul-Chunk-Iso
bash scripts/finetune.sh 0,1 61000 llama2-7b 4 1e-05 4 FT-Mul-Chunk-Iso $save_folder
```

### Unlearning
We have two unlearn splits, size 100 (`unlearn_split=""`) and people 12 (`unlearn_split="_people12"`). To unlearn with the unlearn data `unlearn_data` (`UL-Exact`, `UL-Single`, `UL-Mul`) from the model trained by any `ft_data` (`FT-Single`, `FT-Mul`, `FT-Mul-Chunk`, `FT-Mul-Chunk-Iso`), the scripts of Gradient Ascent (GA) and Task Vector (TV) are

```
bash scripts/ga.sh 0,1 61002 llama2-7b $ft_data $save_folder -1 $unlearn_data  $unlearn_split; 
bash scripts/tv.sh 0,1 61002 llama2-7b $ft_data $save_folder "unlearn_batch" $unlearn_data $unlearn_split; 
done
```

## Useful scripts for unlearning on (Llama2-7B, TOFU+)
### Finetune Models
Here is the script to finetune Llama2-7b on three types of datasets `FT-Single`, `FT-Mul`, `FT-Mul-Chunk`.
```
#FT-Single
bash scripts_tofu/finetune.sh 1,2 61000 llama2-7b 5 1e-05 4 FT-Single $save_folder

#FT-Mul
bash scripts_tofu/finetune.sh 1,2 61000 llama2-7b 5 1e-05 4 FT-Mul $save_folder

#FT-Mul-Chunk
bash scripts_tofu/finetune.sh 1,2 61000 llama2-7b 4 1e-05 4 FT-Mul-Chunk $save_folder
```

### Unlearning
We use the same unlearn split (`unlearn_split=="_forget01"`) as the original TOFU dataset. To unlearn with the unlearn data `unlearn_data` (`UL-Exact`, `UL-Single`, `UL-Mul`) from the model trained by any `ft_data` (`FT-Single`, `FT-Mul`, `FT-Mul-Chunk`), the scripts of Gradient Ascent (GA) and Task Vector (TV) are

```
bash scripts_tofu/ga.sh 0,1 61002 llama2-7b $ft_data $save_folder -1 $unlearn_data  $unlearn_split; 
bash scripts_tofu/tv.sh 0,1 61002 llama2-7b $ft_data $save_folder "unlearn_batch" $unlearn_data $unlearn_split; 
done
```
