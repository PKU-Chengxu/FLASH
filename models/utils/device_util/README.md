# device_util

Simulation utilities for device

## model.json

json file that defines your model hyper-parameters, file for RNN model should be in following format:

```
[
	{
		"layer": layer type string, i.e. 'lstm',
		"in_size": input size of this layer, integer,
		"out_size": output size of this layer,
		"batch_size": batch size
		"seq_len": input length / number steps / sequence length
	},
	...
	{...}
]
```

