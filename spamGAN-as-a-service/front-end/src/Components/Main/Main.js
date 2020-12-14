import React, {Component} from 'react';
import { Input, Upload, message, Layout, Button, Table, Spin, Card } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import axios from 'axios';
import "./Main.css";
import 'antd/es/spin/style/css';
import 'antd/dist/antd.css';


const { TextArea } = Input;
const { Header, Content, Footer } = Layout;


const columns = [
    {
        title: 'Review',
        dataIndex: 'review',
        width: '80%',
    },
    {
        title: 'Result',
        dataIndex: 'result',
        width: '20%',
    }
]


const getRandomuserParams = params => {
    return {
      results: params.pagination.pageSize,
      page: params.pagination.current,
      ...params,
    };
};

class Main extends Component {
   
    state = {
        selectedFile: null,
        review: null,
        loading: true,
        data: [],
        reviews: [],
        pagination: {
            current: 1,
            pageSize: 10,
        },
        showTable: false,
        showCard: false,
        result: null
    }

    fileChangeHandler = (info, event) => {
        if (info.file.status !== 'uploading') {
            console.log(info.file, info.fileList);
        }
        if (info.file.status === 'done') {
            message.success(`${info.file.name} file uploaded successfully`);
        } else if (info.file.status === 'error') {
            message.error(`${info.file.name} file upload failed.`);
        }
    }

    visualizationHandler = () => {
        const url = 'http://data-visualized.s3-website.us-east-2.amazonaws.com/';
        window.open(url, '_blank');
    }

    reviewChangeHandler = (event) => {
        this.setState({review: event.target.value})
    }

    postFileHandler = () => {
        if (this.state.review) {
            this.setState({showCard: true, loading: true})
            axios.post('http://3.139.179.60/singleinference', this.state.review, {timeout: 5 * 60 * 1000})
            .then(response => {
                console.log(response);
                const labels = response.data
                const res = labels[0] === 1 ? 'SPAM' : 'Non-SPAM'
                this.setState({result: res, loading: false})
            });
        } else if (this.state.selectedFile) {
            this.setState({loading: true,
                showTable: true})
            const formData = new FormData(); 
            formData.append( 
                "file", 
                this.state.selectedFile, 
                this.state.selectedFile.name 
            ); 
            axios.post('http://3.139.179.60/inference', formData, {timeout: 5 * 60 * 1000})
            .then(response => {
                console.log(response.data);
                const labels = response.data
                const newData = []
                this.state.reviews.map((comment, idx) => {
                    let dataPoint = {
                        key: idx,
                        review: comment,
                        result: labels[idx] === 1 ? 'SPAM' : 'Non-SPAM'
                    };
                    console.log(labels[idx])
                    newData.push(dataPoint)
                })
                const newPagination = {...this.state.pagination,
                    total: newData.length    
                }
                this.setState({
                    data: newData,
                    loading: false,
                    pagination: newPagination,
                    showTable: true
                })
            });
        }
    }
    
    componentDidMount () {
        const { pagination } = this.state;
        axios.get( 'http://3.139.179.60/ping' )
        .then(response => {
            console.log(response);
        });
    }

    render () {
        return (
            <section id="test" >
                <div className="Test">
                    <Content style={{ maxWidth: '800px', margin: 'auto'}}>
                        <TextArea
                            rows={4}
                            name='review'
                            onChange={this.reviewChangeHandler}
                            disabled={this.state.selectedFile}
                            style={{width: '800px'}} />
                        <p style={{marginTop: "30px"}}><strong>Or you can also choose to upload a file. (Only .txt files are acceptable)</strong> </p>
                        <Upload 
                            accept=".txt" 
                            beforeUpload={file => {
                                const reader = new FileReader();
                                const reviews = []
                                reader.onload = e => {
                                    const lines = e.target.result.split('\n')
                                    console.log(lines)
                                    lines.map(line => reviews.push(line))
                                };
                                reader.readAsText(file);
                                this.setState({selectedFile: file,
                                    reviews: reviews})
                                console.log(reviews.length)

                                // Prevent upload
                                return false;
                            }}
                            onChange={this.fileChangeHandler} 
                            name='file'>
                            <Button icon={<UploadOutlined />} disabled={this.state.review}>Click to Upload</Button>
                        </Upload>
                    </Content>
                    <br />
                    <Button type="primary" shape="round" onClick={this.postFileHandler}>Submit</Button>
                    <br />
                    <br />
                    <Button type="primary" shape="round" onClick={this.visualizationHandler}>Visualization for Training Data</Button>
                    <br />
                    <br />
                    <Table
                        style = {{
                            margin: 'auto',
                            borderRadius: '25px',
                            display: this.state.showTable === false ? 'none' : 'block'
                        }}
                        columns={columns}
                        loading={this.state.loading}
                        dataSource={this.state.data}
                        pagination={this.state.pagination} />                    
                </div>
                <Card title="Result"  
                    style={{ width: 300, 
                             margin: 'auto',
                             marginBottom: '20px',
                             display: this.state.showCard === false ? 'none' : 'block' }}
                    loading={this.state.loading} >
                    <p>{this.state.result}</p>
                </Card>
            </section>
        );
    }
    
}

export default Main;