import torch
import glog
import time

def get_graph_wrapper(cls):
    class GraphWrapper(cls):
        def __init__(self, config):
            super(GraphWrapper, self).__init__(config)
            self.built_graph = False

        def forward(self, *args, **kwargs):
            start = time.time()
            if not self.built_graph:
                self.static_args = args
                self.static_kwargs = kwargs

                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    super(GraphWrapper, self).forward(*self.static_args, **self.static_kwargs)
                torch.cuda.current_stream().wait_stream(s)

                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.static_output = super(GraphWrapper, self).forward(*self.static_args, **self.static_kwargs)

                self.built_graph = True
                glog.info("Built CUDA graph of model.")

            # these two loops take < 1e-4 seconds for llama2
            for i in range(len(args)):
                if isinstance(args[i], torch.Tensor):
                    self.static_args[i].copy_(args[i])
            for kw in kwargs:
                if isinstance(kwargs[kw], torch.Tensor):
                    self.static_kwargs[kw].copy_(kwargs[kw])

            self.graph.replay()
            return self.static_output

        def reset(self):
            if self.built_graph:
                del self.static_args, self.static_kwargs
                self.built_graph = False

    return GraphWrapper
