add_triton_library(DynamicCVPipeline
  AddDynamicCVPipeline.cpp
  AddIfControls.cpp
  LifecycleAnalysis.cpp
  InnerMultibuffer.cpp
  OuterMultiBuffer.cpp

  DEPENDS
  AddDynamicCVPipelinePassIncGen

  LINK_LIBS
  BiShengIRHIVMDialect
  BiShengIRScopeDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
  MLIRSCFDialect
)