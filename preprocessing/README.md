## Preprocessing pipeline
### Usage
This requires slides to be scanned at similar highest magnification (This won't work if there are both 20x and 40x slides).  

- To generate tiles at specified size (size at highest magnification), overlap percentage  
`--write_batch_size` will write batched data, which saves tile I/O costs. 

```
generate_tiles.py --data_dir <root_dat_dir> --tile_size <size_of_tiles_at_highest_magnification> --overlap 
--ts_thres <tissue_threshod --num_ps <number_of_processes_to_spawn> --write_batch_size <write_n_slides_together>
``` 