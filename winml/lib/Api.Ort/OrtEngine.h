#include "iengine.h"

#include "core/session/winml_adapter_c_api.h"

#include <memory>

namespace Windows::AI::MachineLearning {

using UniqueOrtModel = std::unique_ptr<OrtModel, void (*)(OrtModel*)>;

class OnnxruntimeEngine;

class OnnruntimeModel : public Microsoft::WRL::RuntimeClass<
                       Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                       IModel> 
{
public:
	OnnruntimeModel();

    HRESULT RuntimeClassInitialize(_In_ OnnxruntimeEngine* engine, _In_ UniqueOrtModel&& ort_model);
    STDMETHOD(GetAuthor)(const char** out, size_t* len);
    STDMETHOD(GetName)(const char** out, size_t* len);
    STDMETHOD(GetDomain)(const char** out, size_t* len);
    STDMETHOD(GetDescription)(const char** out, size_t* len);
    STDMETHOD(GetVersion)(int64_t* out);
    STDMETHOD(GetModelMetadata)(ABI::Windows::Foundation::Collections::IMapView<HSTRING, HSTRING> ** metadata);
    STDMETHOD(GetInputFeatures)(ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>** features);
    STDMETHOD(GetOutputFeatures)(ABI::Windows::Foundation::Collections::IVectorView<winml::ILearningModelFeatureDescriptor>** features);
    STDMETHOD(CloneModel)(IModel** copy);

private:
    HRESULT EnsureMetadata();

private:
    UniqueOrtModel ort_model_;
    Microsoft::WRL::ComPtr<OnnxruntimeEngine> engine_;

    std::optional<std::unordered_map<std::string, std::string>> metadata_cache_;
};

class OnnxruntimeEngine : public Microsoft::WRL::RuntimeClass<
                       Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>,
                       IEngine>
{
public:
	HRESULT RuntimeClassInitialize();
    STDMETHOD(CreateModel)(_In_ const char* model_path, _In_ size_t len, _Outptr_ IModel** out);
    STDMETHOD(CreateModel)( _In_ void* data, _In_ size_t size, _Outptr_ IModel** out);

	const OrtApi* UseOrtApi();
    const WinmlAdapterApi* UseWinmlAdapterApi();

private:
    const OrtApi* ort_api_ = nullptr;
    const WinmlAdapterApi* winml_adapter_api_ = nullptr;
};

}